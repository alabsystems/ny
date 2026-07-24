// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// HiGHS MIP verification path for FC+ReLU networks. Part of #1763.
// Parallel to smt.rs (ay path); HiGHS is open-source (MIT) with LP tightening.

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_core::{Bound, VerificationResult};
use ny_mip::{
    encode_feedforward, LpTightener, MipBackend, MipConfig, MipResult, MipSolver, SplitUnsatCache,
};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{Network, PhaseBudgetConfig};
use ny_tensor::BoundedTensor;
use std::path::Path;

use super::mip_preprocess::{
    bounded_tensor_to_bounds, convert_intermediate_bounds, extract_linear_relu_params,
    fold_constant_layers, strip_shape_layers, unfold_conv2d_to_linear,
    validate_mip_feedforward_topology,
};
use super::mip_single_hidden::{
    collect_exact_single_hidden_intermediate_bounds, is_single_hidden_linear_relu_linear,
};
use super::output::{format_verification_result_json, verification_result_exit_code};
use super::BetaCrownModel;
use intermediate_bounds::collect_mip_intermediate_bounds;
#[cfg(test)]
use intermediate_bounds::collect_mip_intermediate_bounds_with_deadline;
use warm_start::build_warm_start_vector;

/// Human-readable solver name for a MIP backend (verdict output/diagnostics).
fn backend_name(backend: MipBackend) -> &'static str {
    match backend {
        MipBackend::Ay => "ay",
        MipBackend::AyProc => "ay-proc",
    }
}

/// Verify a sequential FC+ReLU network with the MILP pipeline on the ay
/// backend (SOLVER POLICY: ny-mip docs/SOLVER_POLICY.md).
///
/// This is the VNN-COMP `mip` path: encode the network with tight Big-M ReLUs
/// and solve it exactly (sat_relu, safenlp, malbeware).
/// Reference: designs/2026-03-04-highs-mip-solver-integration.md (historical)
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_with_mip(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    vnnlib: Option<&VnnLibSpec>,
    property: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    timeout: u64,
    warm_start_candidate: Option<&ArrayD<f32>>,
    mip_solver: crate::MipSolverArg,
    json: bool,
) -> Result<()> {
    let backend = mip_solver.mip_backend();
    // MIP verification only supports sequential networks
    let network = match model_net {
        BetaCrownModel::Sequential(net) => net,
        BetaCrownModel::Graph(_) => {
            anyhow::bail!("MIP verification only supports sequential networks (no residual/attention). Use --complete-verifier bab for DAG models.");
        }
    };

    if !json {
        println!(
            "\nRunning MIP verification ({} solver)...",
            backend_name(backend)
        );
    }

    let config = MipConfig {
        backend,
        timeout_secs: timeout as f64,
        lp_tighten: true, // Tighten CROWN-IBP bounds via LP relaxation before MIP (#3218)
        ..Default::default()
    };

    let start = std::time::Instant::now();

    // Unfold Conv2d into an equivalent Linear layer before stripping shape ops;
    // once unfolded, the following Flatten is a no-op. (#3218)
    let mip_network = unfold_conv2d_to_linear(network, input.shape())?;
    // Strip shape-only layers and fold constants for the flat MIP encoding.
    let mip_network = strip_shape_layers(&mip_network);
    let mip_network = fold_constant_layers(&mip_network)?;

    // The encoder inserts ReLU after every non-final Linear.  Validate the
    // original sequence exactly before erasing it into affine parameters; a
    // membership-only Linear/ReLU check can certify a different network.
    validate_mip_feedforward_topology(&mip_network)?;
    let use_exact_single_hidden_fast_path = is_single_hidden_linear_relu_linear(&mip_network);

    // Keep original layer indices for `convert_intermediate_bounds()`. #3864
    // uses plain IBP only for the exact single-hidden fast path.
    let intermediate_bounds = if use_exact_single_hidden_fast_path {
        collect_exact_single_hidden_intermediate_bounds(network, input)?
    } else {
        // General path: budgeted CROWN-IBP falls back to plain IBP after a
        // short preprocessing deadline so HiGHS keeps most of the budget.
        collect_mip_intermediate_bounds(
            network,
            input,
            config.timeout_secs,
            &PhaseBudgetConfig::default(),
        )?
    };

    // Extract weights/dimensions from the structural folded network, but use
    // the exact-f64 bias sidecar produced by `fold_constant_layers`. The f32
    // biases inside `mip_network` exist only to satisfy `LinearLayer`'s storage
    // type and may be rounded; using them for a certified UNSAT would prove a
    // nearby network rather than the original constant-op algebra.
    let (weights, _structural_biases, layer_dims) = extract_linear_relu_params(&mip_network)?;
    let biases = mip_network.exact_biases().to_vec();
    if biases.len() != weights.len() {
        anyhow::bail!(
            "MIP constant-fold bias sidecar mismatch: {} biases for {} Linear layers",
            biases.len(),
            weights.len()
        );
    }
    let input_bounds = bounded_tensor_to_bounds(input)?;
    // Use the original network for index alignment: IBP outputs follow the
    // original layer order, so the stripped network would misalign shape-layer
    // indices and feed the encoder the wrong Big-M bounds.
    let intermediate_bounds_vec = convert_intermediate_bounds(&intermediate_bounds, network)?;

    // LP tightening: tighter Big-M → faster B&B. Ref: α-β-CROWN bounds_core.py:37-92
    let intermediate_bounds_vec = if config.lp_tighten && !use_exact_single_hidden_fast_path {
        let tighten_start = std::time::Instant::now();
        let tighten_config = MipConfig {
            timeout_secs: config.timeout_secs * 0.1, // 10% of budget for LP
            ..config
        };
        let mut tightened = intermediate_bounds_vec;
        let mut total_stable = 0usize;
        let mut total_unstable = 0usize;
        // Progressive: rebuild tightener each layer so layer N uses tightened 0..N-1
        for layer_idx in 0..tightened.len() {
            let tightener = LpTightener::new(
                weights.clone(),
                biases.clone(),
                layer_dims.clone(),
                input_bounds.clone(),
                tightened.clone(),
                tighten_config,
            );
            let (new_bounds, newly_stable) = tightener
                .tighten_layer(layer_idx, &tightened[layer_idx])
                .map_err(|e| {
                    anyhow::anyhow!("LP tightening failed on layer {}: {}", layer_idx, e)
                })?;
            total_unstable += new_bounds
                .iter()
                .filter(|b| b.lower() < 0.0 && b.upper() > 0.0)
                .count();
            total_stable += newly_stable;
            tightened[layer_idx] = new_bounds;
        }
        if !json {
            println!(
                "  LP tightening: {total_stable} neurons fixed stable, {total_unstable} remain unstable ({:.2}s)",
                tighten_start.elapsed().as_secs_f64()
            );
        }
        tightened
    } else {
        intermediate_bounds_vec
    };

    // Subtract preprocessing time from MIP budget to stay within VNN-COMP deadline.
    let preprocessing_elapsed = start.elapsed().as_secs_f64();
    let mip_timeout_secs = (config.timeout_secs - preprocessing_elapsed).max(1.0);
    let mip_config = MipConfig {
        timeout_secs: mip_timeout_secs,
        ..config
    };

    // Solve: handle disjunctive vs conjunctive properties
    let num_outputs = vnnlib.map(|s| s.num_outputs).unwrap_or(1);
    let result = if let Some(spec) = vnnlib {
        if spec.is_disjunction && spec.output_constraint_clauses.len() > 1 {
            // Disjunctive property: solve each clause independently.
            // SAT on ANY clause → Violated. UNSAT on ALL → Verified.
            solve_disjunctive(
                network,
                input,
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &intermediate_bounds_vec,
                spec,
                mip_config,
                num_outputs,
                json,
            )?
        } else {
            // Conjunctive property: single solve, optionally warm-started from PGD (#3865).
            let mut encoder = encode_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &intermediate_bounds_vec,
            )
            .map_err(|e| anyhow::anyhow!("MIP encoding failed: {}", e))?;
            add_vnnlib_constraints(&mut encoder, spec)?;
            let parts = encoder.into_parts();
            let warm_start_cols = warm_start_candidate.and_then(|candidate| {
                build_warm_start_vector(
                    candidate,
                    &weights,
                    &biases,
                    &layer_dims,
                    &intermediate_bounds_vec,
                    parts.num_cols,
                )
            });
            let solver = MipSolver::new(parts, mip_config);
            let mip_result = solver
                .check_feasibility_with_warm_start(warm_start_cols.as_deref())
                .map_err(|e| anyhow::anyhow!("MIP solve failed: {}", e))?;
            // Soundness gate: clamp + independent forward revalidation before
            // claiming Violated. Constraints are the same conjunction fed to the
            // encoder by add_vnnlib_constraints.
            let conjunctive_constraints = conjunctive_constraints_owned(spec);
            let was_sat = matches!(&mip_result, MipResult::Sat { .. });
            let revalidated = map_mip_result_revalidated(
                mip_result,
                network,
                input,
                &conjunctive_constraints,
                num_outputs,
            );
            if was_sat && !matches!(revalidated, VerificationResult::Violated { .. }) {
                // Solver-tolerance witness: re-solve with a violation slack for
                // a robust one (see retry_with_violation_slack).
                retry_with_violation_slack(
                    network,
                    input,
                    &weights,
                    &biases,
                    &layer_dims,
                    &input_bounds,
                    &intermediate_bounds_vec,
                    &conjunctive_constraints,
                    mip_config,
                    num_outputs,
                )
                .unwrap_or(revalidated)
            } else {
                revalidated
            }
        }
    } else {
        // Non-VNNLIB path: threshold defines safety property (output >= threshold).
        // Unsafe region: output < threshold. In LP/MIP, approximate with output <= threshold.
        let mut encoder = encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds_vec,
        )
        .map_err(|e| anyhow::anyhow!("MIP encoding failed: {}", e))?;
        encoder
            .constrain_output_leq_const(0, threshold as f64)
            .map_err(|e| anyhow::anyhow!("constraint failed: {}", e))?;
        let parts = encoder.into_parts();
        let solver = MipSolver::new(parts, mip_config);
        // Non-VNNLIB path: no PGD candidate available, warm-start not applicable.
        let mip_result = solver
            .check_feasibility_with_warm_start(None)
            .map_err(|e| anyhow::anyhow!("MIP solve failed: {}", e))?;
        // Soundness gate: the safety property is output[0] >= threshold, so the
        // unsafe region is output[0] < threshold. Revalidate the witness against
        // an equivalent LessThanConst(0, threshold) constraint.
        let threshold_constraint = vec![OutputConstraint::LessThanConst(0, threshold as f64)];
        map_mip_result_revalidated(
            mip_result,
            network,
            input,
            &threshold_constraint,
            num_outputs,
        )
    };
    let elapsed = start.elapsed();

    // Output results
    print_result(
        &result, property, epsilon, threshold, elapsed, backend, json,
    )?;

    let exit_code = verification_result_exit_code(&result);
    if exit_code != crate::commands::verify::exit_codes::VERIFIED && !super::output::is_capturing()
    {
        std::process::exit(exit_code);
    }

    Ok(())
}

/// Owned copy of the conjunctive constraint list fed to `add_vnnlib_constraints`.
///
/// Mirrors `add_vnnlib_constraints`' selection logic so the revalidation gate
/// re-checks exactly the constraints the encoder asserted. Prefers the flattened
/// `output_constraint_clauses` (non-disjunctive specs may still carry a single
/// clause there) and falls back to `output_constraints`.
fn conjunctive_constraints_owned(spec: &VnnLibSpec) -> Vec<OutputConstraint> {
    if !spec.output_constraint_clauses.is_empty() {
        spec.output_constraint_clauses
            .iter()
            .flatten()
            .cloned()
            .collect()
    } else {
        spec.output_constraints.clone()
    }
}

/// Add VNNLIB output constraints to the MIP encoder (conjunctive only).
///
/// Disjunctive specs are handled by `solve_disjunctive` upstream.
fn add_vnnlib_constraints(encoder: &mut ny_mip::MipEncoder, spec: &VnnLibSpec) -> Result<()> {
    let constraints = if !spec.output_constraint_clauses.is_empty() {
        spec.output_constraint_clauses
            .iter()
            .flatten()
            .collect::<Vec<_>>()
    } else {
        spec.output_constraints.iter().collect::<Vec<_>>()
    };

    for constraint in constraints {
        encode_output_constraint(encoder, constraint)?;
    }
    Ok(())
}

/// Solve disjunctive VNNLIB property by solving each clause independently.
///
/// Strategy: for each clause, encode a separate MIP and solve.
/// - If ANY clause is SAT → the overall property is VIOLATED
/// - If ALL clauses are certified UNSAT → the overall property is VERIFIED
/// - If any clause times out and none is SAT → TIMEOUT
///
/// Reference: alpha-beta-CROWN solves disjunctive properties the same way
/// (one MIP per clause, early exit on SAT).
#[allow(clippy::too_many_arguments)]
fn solve_disjunctive(
    network: &Network,
    input: &BoundedTensor,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    layer_dims: &[usize],
    input_bounds: &[Bound],
    intermediate_bounds: &[Vec<Bound>],
    spec: &VnnLibSpec,
    config: MipConfig,
    num_outputs: usize,
    json: bool,
) -> Result<VerificationResult> {
    let num_clauses = spec.output_constraint_clauses.len();
    if !json {
        println!(
            "  Disjunctive property: {} clauses, solving independently...",
            num_clauses
        );
    }

    let mut had_timeout = false;
    // An exact solver status without independently checked infeasibility
    // evidence is not a proof. Such a clause must block Verified just like any
    // other undecided clause, but retrying the identical solve cannot upgrade
    // its evidence, so remember it separately from timeouts.
    let mut had_uncertified_unsat = false;
    // Tracks clauses where the MIP found a feasible unsafe point but the witness
    // failed in-box revalidation. The clause's unsafe region IS reachable, so we
    // must NOT conclude Verified — only the concrete witness is unconfirmed.
    let mut had_unconfirmed_sat = false;
    let overall_start = std::time::Instant::now();
    let overall_budget = std::time::Duration::from_secs_f64(config.timeout_secs);

    // Progressive multi-round schedule (#malbeware-mip-budget): the first pass
    // gives every clause `remaining / remaining_clauses`; clauses that hit that
    // slice (Timeout/Error) are RETRIED while overall budget remains, with the
    // slice recomputed over the (much smaller) undecided set. Measured on
    // malbeware 4-25 eps-3 (24 clauses, ~42s MIP slice): ~20 easy clauses close
    // in ~0.4s each, so a hard clause's slice grows from ~1.8s (round 1) to
    // 20s+ (round 2) — the single-pass schedule instead returned `timeout`
    // with >20s of granted budget unused. SOUND: per-clause verdict semantics
    // are unchanged (Unsat still requires proven infeasibility on EVERY
    // clause; a retry only grants a clause more solver time), and any Sat
    // still passes the in-box revalidation gate before being emitted.
    // Round cap: pathological non-progress can at most burn the granted MIP
    // budget, but keep a hard cap so a zero-second-slice loop cannot spin.
    const MAX_CLAUSE_ROUNDS: usize = 4;

    // Encode the network ONCE; per clause, stamp the clause's output
    // constraint onto a clone. The base encode re-scans the dense unfolded
    // weight matrices (252M f64 for malbeware 16-25), so re-encoding per
    // clause cost ~seconds x 24 clauses; the clone copies only the built
    // sparse IR. Identical formulation: `encode_feedforward` is deterministic
    // in its inputs, which do not change across clauses.
    let base_encoder = encode_feedforward(
        weights,
        biases,
        layer_dims,
        input_bounds,
        intermediate_bounds,
    )
    .map_err(|e| anyhow::anyhow!("MIP encoding failed: {}", e))?;

    let mut pending: Vec<usize> = (0..num_clauses).collect();
    // One certified-UNSAT phase-split memo per clause, living ACROSS retry
    // rounds: a clause abandoned at 15-of-16 certified-Unsat subproblems
    // re-races only the open one instead of starting from zero. The memo is
    // keyed by a full problem fingerprint inside ny-mip (fail-closed: any
    // re-encode drift clears it); the deterministic base_encoder clone +
    // clause stamping above is what makes the fingerprint match across
    // rounds.
    let mut split_caches: Vec<SplitUnsatCache> = (0..num_clauses)
        .map(|_| SplitUnsatCache::default())
        .collect();
    'rounds: for round in 0..MAX_CLAUSE_ROUNDS {
        if pending.is_empty() {
            break;
        }
        if round > 0 {
            let remaining = overall_budget.saturating_sub(overall_start.elapsed());
            // A retry round needs a meaningful slice to make progress; stop
            // once the tail budget is exhausted (< 1s per pending clause).
            if remaining.as_secs_f64() < pending.len() as f64 {
                break;
            }
            if !json {
                println!(
                    "  Retry round {}: {} timed-out clause(s), {:.1}s budget remaining",
                    round + 1,
                    pending.len(),
                    remaining.as_secs_f64()
                );
            }
        }
        let round_pending = std::mem::take(&mut pending);
        let round_total = round_pending.len();

        for (pos, &clause_idx) in round_pending.iter().enumerate() {
            let clause = &spec.output_constraint_clauses[clause_idx];
            // Adaptive per-clause timeout: remaining budget / remaining clauses
            // in this round. Prevents early clauses from consuming the budget.
            let elapsed = overall_start.elapsed();
            if elapsed >= overall_budget {
                // Out of budget: everything not yet decided stays pending.
                pending.extend(round_pending[pos..].iter().copied());
                break 'rounds;
            }
            let remaining_secs = overall_budget
                .checked_sub(elapsed)
                .expect("guarded above: elapsed < overall_budget")
                .as_secs_f64();
            let remaining_clauses = (round_total - pos).max(1) as f64;
            let clause_timeout = remaining_secs / remaining_clauses;
            let clause_config = MipConfig {
                timeout_secs: clause_timeout,
                ..config
            };

            let mut encoder = base_encoder.clone();

            for constraint in clause {
                encode_output_constraint(&mut encoder, constraint)?;
            }

            let parts = encoder.into_parts();
            let solver = MipSolver::new(parts, clause_config);
            let mip_result = solver
                .check_feasibility_cached(None, &mut split_caches[clause_idx])
                .map_err(|e| anyhow::anyhow!("MIP solve failed on clause {}: {}", clause_idx, e))?;

            match &mip_result {
                MipResult::Sat { .. } => {
                    // Soundness gate: clamp + independent forward revalidation against
                    // THIS clause's constraints (the conjunction defining the disjunct).
                    // Only a confirmed in-box violation is emitted as Violated; an
                    // unconfirmed witness is treated as "no violation on this clause"
                    // so we keep probing remaining clauses (one may genuinely violate).
                    let revalidated =
                        map_mip_result_revalidated(mip_result, network, input, clause, num_outputs);
                    match revalidated {
                        VerificationResult::Violated { .. } => {
                            if !json {
                                println!(
                                    "  Clause {}/{}: SAT (counterexample confirmed in-box)",
                                    clause_idx + 1,
                                    num_clauses
                                );
                            }
                            return Ok(revalidated);
                        }
                        _ => {
                            // Solver-tolerance witness: re-solve this clause with a
                            // violation slack for a robust one.
                            if let Some(v) = retry_with_violation_slack(
                                network,
                                input,
                                weights,
                                biases,
                                layer_dims,
                                input_bounds,
                                intermediate_bounds,
                                clause,
                                clause_config,
                                num_outputs,
                            ) {
                                if !json {
                                    println!(
                                        "  Clause {}/{}: SAT (robust witness via violation-slack retry)",
                                        clause_idx + 1,
                                        num_clauses
                                    );
                                }
                                return Ok(v);
                            }
                            if !json {
                                println!(
                                    "  Clause {}/{}: SAT but failed in-box revalidation (cannot conclude verified)",
                                    clause_idx + 1,
                                    num_clauses
                                );
                            }
                            // The clause's unsafe region is reachable per the MIP, so
                            // concluding Verified here would be UNSOUND. Mark the run
                            // inconclusive (Unknown/Timeout) rather than safe. Not
                            // retried: a re-solve reproduces the same tolerance
                            // witness (the slack retry above already probed for a
                            // robust one).
                            had_unconfirmed_sat = true;
                        }
                    }
                }
                MipResult::Unsat { certified: true } => {
                    if !json {
                        println!(
                            "  Clause {}/{}: UNSAT (certified)",
                            clause_idx + 1,
                            num_clauses,
                        );
                    }
                }
                MipResult::Unsat { certified: false } => {
                    had_uncertified_unsat = true;
                    if !json {
                        println!(
                            "  Clause {}/{}: UNSAT without checked certificate (cannot conclude verified)",
                            clause_idx + 1,
                            num_clauses,
                        );
                    }
                }
                MipResult::Timeout => {
                    if !json {
                        println!(
                            "  Clause {}/{}: TIMEOUT ({:.1}s slice; will retry with the tail budget)",
                            clause_idx + 1,
                            num_clauses,
                            clause_timeout
                        );
                    }
                    pending.push(clause_idx);
                }
                MipResult::Error(msg) => {
                    if !json {
                        println!(
                            "  Clause {}/{}: ERROR ({})",
                            clause_idx + 1,
                            num_clauses,
                            msg
                        );
                    }
                    pending.push(clause_idx);
                }
            }
        }
    }
    if !pending.is_empty() {
        had_timeout = true;
    }

    let output_bounds =
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); num_outputs];
    match disjunctive_proof_status(had_unconfirmed_sat, had_uncertified_unsat, had_timeout) {
        DisjunctiveProofStatus::Unknown(reason) => Ok(VerificationResult::Unknown {
            provenance: Default::default(),
            bounds: output_bounds,
            reason: ny_core::UnknownReason::SmtUnknown {
                solver_reason: Some(reason.to_string()),
            },
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        }),
        DisjunctiveProofStatus::Timeout => Ok(VerificationResult::Timeout {
            provenance: Default::default(),
            partial_bounds: Some(output_bounds),
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        }),
        DisjunctiveProofStatus::Verified => Ok(VerificationResult::Verified {
            provenance: Default::default(),
            output_bounds,
            proof: None,
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DisjunctiveProofStatus {
    Unknown(&'static str),
    Timeout,
    Verified,
}

fn disjunctive_proof_status(
    had_unconfirmed_sat: bool,
    had_uncertified_unsat: bool,
    had_timeout: bool,
) -> DisjunctiveProofStatus {
    if had_unconfirmed_sat {
        DisjunctiveProofStatus::Unknown("disjunctive MIP sat witness failed in-box revalidation")
    } else if had_uncertified_unsat {
        DisjunctiveProofStatus::Unknown("disjunctive MIP UNSAT lacked a checked certificate")
    } else if had_timeout {
        DisjunctiveProofStatus::Timeout
    } else {
        DisjunctiveProofStatus::Verified
    }
}

fn encode_output_constraint(
    encoder: &mut ny_mip::MipEncoder,
    constraint: &OutputConstraint,
) -> Result<()> {
    let r = match constraint {
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
            encoder.constrain_output_leq(*i, *j)
        }
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            encoder.constrain_output_geq(*i, *j)
        }
        OutputConstraint::LessEqConst(i, c) | OutputConstraint::LessThanConst(i, c) => {
            encoder.constrain_output_leq_const(*i, *c)
        }
        OutputConstraint::GreaterEqConst(i, c) | OutputConstraint::GreaterThanConst(i, c) => {
            encoder.constrain_output_geq_const(*i, *c)
        }
        _ => return Err(anyhow::anyhow!("unsupported OutputConstraint variant")),
    };
    r.map_err(|e| anyhow::anyhow!("output constraint encoding failed: {}", e))
}

/// Print verification result in human-readable or JSON format.
pub(super) fn print_result(
    result: &VerificationResult,
    property: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    elapsed: std::time::Duration,
    backend: MipBackend,
    json: bool,
) -> Result<()> {
    if json {
        let method = match backend {
            MipBackend::Ay => "mip-ay",
            MipBackend::AyProc => "mip-ay-proc",
        };
        let rendered =
            format_verification_result_json(result, property, epsilon, threshold, elapsed, method)?;
        super::output::emit_competition_json(&rendered);
        return Ok(());
    }
    match result {
        VerificationResult::Verified { .. } => println!("Status: VERIFIED (safe)"),
        VerificationResult::Violated {
            counterexample,
            output,
            ..
        } => {
            println!("Status: VIOLATED (unsafe)");
            println!("Counterexample input: {:?}", counterexample);
            if let Some(out) = output.first() {
                println!("Counterexample output: {}", out);
            }
        }
        VerificationResult::Timeout { .. } => println!("Status: TIMEOUT"),
        VerificationResult::Unknown { reason, .. } => {
            println!("Status: UNKNOWN");
            println!("Reason: {}", reason);
        }
    }
    println!("Method: MIP ({} solver)", backend_name(backend));
    println!("Time elapsed: {:.2}s", elapsed.as_secs_f64());
    Ok(())
}

/// Map a non-SAT MIP result (Unsat / Timeout / Error) to a `VerificationResult`.
///
/// The SAT arm is handled separately by [`revalidate_mip_witness`], which clamps
/// the witness into the VNN-LIB box and re-checks the spec with an independent
/// forward pass before claiming `Violated`. `result` MUST NOT be `Sat` here.
fn map_mip_nonsat_result(result: MipResult, num_outputs: usize) -> VerificationResult {
    let bounds = || vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); num_outputs];
    match result {
        // Only independently checked exact evidence (Farkas or case-split)
        // may turn solver infeasibility into a verifier verdict.
        MipResult::Unsat { certified: true } => {
            tracing::info!("MIP UNSAT admitted with verified exact certificate");
            VerificationResult::Verified {
                provenance: Default::default(),
                output_bounds: bounds(),
                proof: None,
                actual_method: Some(ny_core::MethodUsed::MipHiGHS),
            }
        }
        MipResult::Unsat { certified: false } => VerificationResult::Unknown {
            provenance: Default::default(),
            bounds: bounds(),
            reason: ny_core::UnknownReason::SmtUnknown {
                solver_reason: Some("MIP UNSAT lacked a checked certificate".to_string()),
            },
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        },
        MipResult::Sat { .. } => {
            // Defensive: SAT must be revalidated via revalidate_mip_witness, never
            // emitted raw. Treat an unexpected SAT here as Unknown (sound) rather
            // than fabricating an un-revalidated counterexample.
            VerificationResult::Unknown {
                provenance: Default::default(),
                bounds: bounds(),
                reason: ny_core::UnknownReason::SmtUnknown {
                    solver_reason: Some(
                        "MIP SAT reached map_mip_nonsat_result without revalidation".to_string(),
                    ),
                },
                actual_method: Some(ny_core::MethodUsed::MipHiGHS),
            }
        }
        MipResult::Timeout => VerificationResult::Timeout {
            provenance: Default::default(),
            partial_bounds: Some(bounds()),
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        },
        MipResult::Error(msg) => VerificationResult::Unknown {
            provenance: Default::default(),
            bounds: bounds(),
            reason: ny_core::UnknownReason::SmtUnknown {
                solver_reason: Some(msg),
            },
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        },
    }
}

/// Margin a concrete output has against an unsafe-region constraint.
///
/// Positive means the constraint is satisfied (output is in the unsafe region)
/// with that much slack; negative/zero means it does not (strictly) hold. OOB
/// indices map to `-inf` so a malformed constraint can never confirm a violation.
/// Mirrors the margin convention in `verify/disjunctive_pgd.rs`.
pub(super) fn mip_constraint_margin(constraint: &OutputConstraint, output: &ArrayD<f32>) -> f32 {
    let at = |i: usize| output.iter().nth(i).copied();
    match constraint {
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            match (at(*i), at(*j)) {
                (Some(yi), Some(yj)) => yi - yj,
                _ => f32::NEG_INFINITY,
            }
        }
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
            match (at(*i), at(*j)) {
                (Some(yi), Some(yj)) => yj - yi,
                _ => f32::NEG_INFINITY,
            }
        }
        OutputConstraint::GreaterEqConst(i, c) | OutputConstraint::GreaterThanConst(i, c) => {
            at(*i).map_or(f32::NEG_INFINITY, |y| y - *c as f32)
        }
        OutputConstraint::LessEqConst(i, c) | OutputConstraint::LessThanConst(i, c) => {
            at(*i).map_or(f32::NEG_INFINITY, |y| *c as f32 - y)
        }
        _ => f32::NEG_INFINITY, // unknown variant cannot confirm a violation
    }
}

/// Margin guard absorbing f64->f32 cast drift between the solver's relaxation
/// and the independent CPU forward pass. A wrong VNN-COMP verdict is -150;
/// a timeout/Unknown is not — so borderline SAT claims are demoted, not emitted.
/// Same value as `verify/disjunctive_pgd.rs::re_evaluate_and_confirm`.
const REVALIDATION_MARGIN_EPS: f32 = 1e-5;

/// Descending violation-slack sweep. A larger delta yields a more robust witness
/// (one that survives f32/f64/ORT re-evaluation), but the STRENGTHENED problem is
/// only feasible when EVERY strengthened constraint still has reachable headroom.
/// A single fixed delta demoted genuine boundary-SAT witnesses whenever the
/// TIGHTEST constraint's headroom fell below it: on sat_v33_c140 the `Y_0 >= 1.0`
/// constraint's reachable max is only ~1 + 6.2e-6, so the uniform 1e-5 slack made
/// `Y_0 >= 1.00001` infeasible — independent of how far `Y_1` could be pushed below
/// 0 — and the real boundary witness (`Y_1` reachable a few e-6 below 0) was lost.
/// Sweeping downward and taking the FIRST (largest, most robust) delta whose
/// strengthened solution re-validates recovers these. The floor (5e-7) stays above
/// the measured ~5e-7 f32 forward drift so any accepted witness clears the
/// zero-tolerance revalidation gate. SOUND at every delta: the strengthened unsafe
/// region ⊆ the original, so a solution genuinely violates the original property,
/// and the witness still passes the independent zero-tolerance revalidation.
const VIOLATION_SLACKS: [f64; 5] = [1e-5, 5e-6, 2e-6, 1e-6, 5e-7];

/// Whether a constraint is a shiftable const-threshold comparison.
fn is_shiftable_const(c: &OutputConstraint) -> bool {
    use OutputConstraint as OC;
    matches!(
        c,
        OC::GreaterEqConst(..)
            | OC::GreaterThanConst(..)
            | OC::LessEqConst(..)
            | OC::LessThanConst(..)
    )
}

/// Recover a robust counterexample after a MIP `Sat` witness failed exact in-box
/// revalidation (a solver-tolerance artifact: the LP-feasible point can miss the true
/// forward by ~1e-6). Re-solve the SAME property strengthened by a violation slack so
/// any solution clears the zero-tolerance revalidation gate. SOUND: the strengthened
/// unsafe region ⊆ the original, so a solution genuinely violates it, and the returned
/// witness still passes the independent revalidation in `map_mip_result_revalidated`;
/// infeasibility proves nothing and the caller keeps its conservative outcome. See
/// [`VIOLATION_SLACKS`] for why the slack is swept per-constraint rather than uniform.
#[allow(clippy::too_many_arguments)]
fn retry_with_violation_slack(
    network: &Network,
    input: &BoundedTensor,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    layer_dims: &[usize],
    input_bounds: &[Bound],
    intermediate_bounds: &[Vec<Bound>],
    constraints: &[OutputConstraint],
    config: MipConfig,
    num_outputs: usize,
) -> Option<VerificationResult> {
    use OutputConstraint as OC;
    if !constraints.iter().any(is_shiftable_const) {
        return None; // nothing shiftable — no robustness to gain
    }

    let solve = |cs: &[OC], secs: f64| -> Option<MipResult> {
        let mut enc = encode_feedforward(
            weights,
            biases,
            layer_dims,
            input_bounds,
            intermediate_bounds,
        )
        .ok()?;
        for c in cs {
            encode_output_constraint(&mut enc, c).ok()?;
        }
        let cfg = MipConfig {
            timeout_secs: secs,
            ..config
        };
        MipSolver::new(enc.into_parts(), cfg)
            .check_feasibility()
            .ok()
    };

    // Budget: half the remaining time, split across an initial diagnostic solve plus
    // the delta sweep, so the retry can never overrun the wall clock.
    let budget = (config.timeout_secs * 0.5).max(4.0);
    let per_try = (budget / (VIOLATION_SLACKS.len() + 1) as f64).max(1.0);

    // Diagnose WHICH constraints miss the independent real forward. Strengthening a
    // constraint that already holds with margin only shrinks the feasible set on that
    // (often tight) axis — e.g. `Y_0 >= 1.0`'s reachable headroom is ~6.2e-6 on
    // sat_v33_c140, so a uniform slack over ALL constraints made `Y_0 >= 1+delta`
    // infeasible independent of how far `Y_1` could move. Strengthen exactly the
    // constraints the witness missed; leave the satisfied ones at their thresholds.
    let mut failing = vec![false; constraints.len()];
    if let Some(MipResult::Sat { input_values, .. }) = solve(constraints, per_try) {
        let clamped = clamp_witness_to_box(&input_values, input);
        if let Ok(out) = independent_mip_forward(network, &clamped) {
            for (idx, c) in constraints.iter().enumerate() {
                if is_shiftable_const(c) && mip_constraint_margin(c, &out) < REVALIDATION_MARGIN_EPS
                {
                    failing[idx] = true;
                }
            }
        }
    }
    // Fall back to strengthening every const constraint if the diagnostic solve did not
    // pinpoint a failing one (preserves the original all-constraint behavior).
    let strengthen_all = !failing.iter().any(|&f| f);
    let strengthen = |delta: f64| -> Vec<OC> {
        constraints
            .iter()
            .enumerate()
            .map(|(idx, c)| {
                if !(strengthen_all || failing[idx]) {
                    return c.clone();
                }
                match c {
                    OC::GreaterEqConst(i, k) => OC::GreaterEqConst(*i, k + delta),
                    OC::GreaterThanConst(i, k) => OC::GreaterThanConst(*i, k + delta),
                    OC::LessEqConst(i, k) => OC::LessEqConst(*i, k - delta),
                    OC::LessThanConst(i, k) => OC::LessThanConst(*i, k - delta),
                    other => other.clone(),
                }
            })
            .collect()
    };

    for &delta in &VIOLATION_SLACKS {
        let strengthened = strengthen(delta);
        if strengthened == constraints {
            continue; // this delta shifted nothing (shouldn't happen post-guard)
        }
        tracing::warn!(
            "violation-slack retry: delta {delta:.0e} (strengthen_all={strengthen_all})"
        );
        let Some(mip_result) = solve(&strengthened, per_try) else {
            continue;
        };
        tracing::warn!("violation-slack retry outcome (delta {delta:.0e}): {mip_result:?}");
        // Revalidate against the ORIGINAL constraints — the property being scored.
        if let v @ VerificationResult::Violated { .. } =
            map_mip_result_revalidated(mip_result, network, input, constraints, num_outputs)
        {
            tracing::warn!("violation-slack retry recovered a robust witness (delta {delta:.0e})");
            return Some(v);
        }
    }
    None
}

/// Whether the exact f64 reference forward confirms the (f32-quantized, in-box)
/// witness violates every constraint under exact SMT-LIB semantics.
/// `Err(reason)` when the f64 path is unavailable for this network (the caller
/// keeps the conservative demotion then).
fn f64_forward_confirms(
    network: &Network,
    clamped: &ArrayD<f32>,
    constraints: &[OutputConstraint],
) -> std::result::Result<bool, String> {
    let layers64 = ny_propagate::convert_network_to_f64(network.layers())
        .map_err(|e| format!("layer conversion failed: {e}"))?;
    let input64 = clamped.mapv(f64::from);
    let out64 = ny_propagate::evaluate_network_f64(&layers64, &input64)
        .map_err(|e| format!("f64 forward failed: {e}"))?;
    let value_at = |i: usize| -> Option<f64> { out64.iter().nth(i).copied() };
    Ok(constraints.iter().all(|c| {
        use OutputConstraint as OC;
        match c {
            OC::LessEq(i, j) => {
                matches!((value_at(*i), value_at(*j)), (Some(a), Some(b)) if a <= b)
            }
            OC::LessThan(i, j) => {
                matches!((value_at(*i), value_at(*j)), (Some(a), Some(b)) if a < b)
            }
            OC::GreaterEq(i, j) => {
                matches!((value_at(*i), value_at(*j)), (Some(a), Some(b)) if a >= b)
            }
            OC::GreaterThan(i, j) => {
                matches!((value_at(*i), value_at(*j)), (Some(a), Some(b)) if a > b)
            }
            OC::LessEqConst(i, k) => value_at(*i).is_some_and(|a| a <= *k),
            OC::LessThanConst(i, k) => value_at(*i).is_some_and(|a| a < *k),
            OC::GreaterEqConst(i, k) => value_at(*i).is_some_and(|a| a >= *k),
            OC::GreaterThanConst(i, k) => value_at(*i).is_some_and(|a| a > *k),
            // Fail closed on any future constraint variant (#4375 pattern).
            _ => false,
        }
    }))
}

fn independent_mip_forward(network: &Network, candidate: &ArrayD<f32>) -> Result<ArrayD<f32>> {
    // Exact concrete forward, NOT IBP `.lower()` of a point box (completes
    // bd68815): IBP's ULP-outward rounding on big-M-scale nets exceeds
    // REVALIDATION_MARGIN_EPS and demoted genuine MIP witnesses (sat_relu).
    let input_bounds = BoundedTensor::concrete(candidate.clone())?;
    let output = network.propagate_concrete_point(&input_bounds, None)?;
    Ok(output.center())
}

/// Clamp a raw solver witness into the VNN-LIB input box.
///
/// Casts each raw f64 to f32 FIRST, then clamps into `[lower, upper]`, so the
/// returned witness is exactly the bytes the organizer's onnxruntime will read
/// AND is guaranteed inside the box even if the f64->f32 cast nudged a coord
/// out. The result is reshaped to `input.shape()` for the forward pass.
pub(super) fn clamp_witness_to_box(raw_input: &[f64], input: &BoundedTensor) -> ArrayD<f32> {
    let lower = input.lower();
    let upper = input.upper();
    let n = lower.len();
    let mut clamped = Vec::with_capacity(n);
    let mut lo_it = lower.iter();
    let mut hi_it = upper.iter();
    for k in 0..n {
        let lo = lo_it.next().copied().unwrap_or(f32::NEG_INFINITY);
        let hi = hi_it.next().copied().unwrap_or(f32::INFINITY);
        // raw_input matches the flattened input box order; if the solver returned
        // fewer values than the box (shouldn't happen), pad with the lower bound.
        let raw = raw_input.get(k).copied().unwrap_or(lo as f64) as f32;
        // clamp() panics if lo > hi; guard degenerate/NaN bounds defensively.
        let v = if lo <= hi {
            raw.clamp(lo, hi)
        } else {
            lo // degenerate box: collapse to lower
        };
        clamped.push(v);
    }
    ArrayD::from_shape_vec(IxDyn(lower.shape()), clamped)
        .expect("clamped witness length equals input box length by construction")
}

/// Clamp a raw MIP/SMT witness into the VNN-LIB box, re-validate it with an
/// independent forward pass through the ORIGINAL network, and emit `Violated`
/// ONLY if the spec is still violated in-box (with an epsilon margin guard).
///
/// This is the soundness gate for every MIP/SMT `Sat`: the organizer re-runs our
/// counterexample through onnxruntime on the eval inputs, rejecting any witness
/// that is out-of-box or that the f64->f32 cast moved off the violation. A wrong
/// verdict is -150; demoting a borderline witness to `Unknown` costs only a
/// timeout-equivalent. We therefore:
///   1. CLAMP each input into `[input.lower(), input.upper()]` (no-op for an
///      already in-box witness, so genuine violations are preserved).
///   2. INDEPENDENT FORWARD through `network` (engine=None), NOT `mip_network`.
///   3. RE-CHECK the spec: every constraint must hold with margin >=
///      `REVALIDATION_MARGIN_EPS`. If so, emit `Violated` with the CLAMPED input
///      and the RE-EVALUATED output (not the solver's relaxed output). Otherwise
///      demote to `Unknown` (sound — never claim a sat we cannot back).
fn revalidate_mip_witness(
    network: &Network,
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
        // No spec constraints means no violation can be confirmed.
        return unknown("MIP sat had no output constraints to revalidate against");
    }

    // 1. Clamp the raw witness into the VNN-LIB box.
    let clamped = clamp_witness_to_box(raw_input, input);

    // 2. Independent forward pass through the ORIGINAL network (engine=None).
    let revalidated = match independent_mip_forward(network, &clamped) {
        Ok(out) => out,
        Err(e) => {
            tracing::warn!("MIP sat witness revalidation forward pass failed: {e}");
            return unknown("MIP sat witness revalidation forward pass failed");
        }
    };

    // 3. Re-check the spec under exact SMT-LIB semantics on the EXACT concrete
    // forward (check_unsafe_counterexample: strict `<`/`>` fail at equality,
    // non-strict `<=`/`>=` accept it). Deliberately NO extra epsilon guard:
    // SAT-encoded nets (sat_relu) construct satisfying assignments with
    // margins of exactly 0.0 or a few ULPs — a blanket eps demoted every real
    // witness the solver found in seconds. Cross-implementation robustness is
    // arbitrated downstream by the trusted-ORT vnncomp gate, which re-runs the
    // witness through real ONNX Runtime before any `sat` is scored (worst case
    // it downgrades to unknown — never a wrong verdict). Sub-eps margins are
    // still logged for diagnosability.
    let mut confirmed = super::verify::check_unsafe_counterexample(&revalidated, constraints);
    if confirmed {
        let min_margin = constraints
            .iter()
            .map(|c| mip_constraint_margin(c, &revalidated))
            .fold(f32::INFINITY, f32::min);
        if min_margin < REVALIDATION_MARGIN_EPS {
            tracing::info!(
                "MIP sat witness confirmed with sub-eps margin {min_margin:.3e} \
                 (< {REVALIDATION_MARGIN_EPS:.1e}); trusted-ORT gate will arbitrate"
            );
        }
    } else {
        // f64 rescue (winner parity: double_fp): SAT-encoded nets construct
        // their violations in real arithmetic; ny's f32 forward can miss by a
        // few ULPs (measured -1.9e-6 on sat_v100) where the faithful f64
        // forward confirms. Emit the sat — the trusted-ORT vnncomp gate still
        // re-runs the witness through real f32 ONNX Runtime before it is
        // scored (worst case a downgrade to unknown, never a wrong verdict).
        match f64_forward_confirms(network, &clamped, constraints) {
            Ok(true) => {
                tracing::warn!(
                    "MIP sat witness confirmed by the f64 reference forward (f32 forward \
                     missed by ULPs); trusted-ORT gate will arbitrate"
                );
                confirmed = true;
            }
            Ok(false) => {
                tracing::warn!("f64 reference forward also rejects the witness");
            }
            Err(reason) => {
                tracing::warn!("f64 rescue unavailable: {reason}");
            }
        }
    }

    if confirmed {
        VerificationResult::Violated {
            provenance: Default::default(),
            counterexample: clamped.iter().copied().collect(),
            output: revalidated.iter().copied().collect(),
            details: None,
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        }
    } else {
        // Diagnostics: how far the clamp moved the raw witness, and how close
        // the clamped point still is to violating (the binding margin). A
        // hair-negative margin points at solver-tolerance boundary witnesses;
        // a grossly negative one at an encoding/precision divergence.
        let max_displacement = clamped
            .iter()
            .zip(raw_input.iter())
            .map(|(c, r)| (f64::from(*c) - r).abs())
            .fold(0.0_f64, f64::max);
        let min_margin = constraints
            .iter()
            .map(|c| mip_constraint_margin(c, &revalidated))
            .fold(f32::INFINITY, f32::min);
        let per_constraint: Vec<String> = constraints
            .iter()
            .map(|c| {
                format!(
                    "{c:?}: margin={:.6e} strict={} unsafe_ok={}",
                    mip_constraint_margin(c, &revalidated),
                    c.is_strict(),
                    super::verify::check_unsafe_counterexample(
                        &revalidated,
                        std::slice::from_ref(c)
                    )
                )
            })
            .collect();
        tracing::warn!(
            "MIP sat witness failed in-box revalidation (clamped to box, spec no longer violated); \
             demoting to Unknown [max clamp displacement {max_displacement:.3e}, \
             min constraint margin at clamped point {min_margin:.3e}, \
             required >= {REVALIDATION_MARGIN_EPS:.1e}; per-constraint: {}]",
            per_constraint.join("; ")
        );
        unknown("MIP sat witness failed in-box revalidation")
    }
}

/// Map a MIP result to a `VerificationResult`, revalidating any `Sat` witness.
///
/// For `Sat`, the witness is clamped into the input box and re-checked with an
/// independent forward pass through the ORIGINAL `network` before emitting
/// `Violated`; an unconfirmed witness is demoted to `Unknown`. All other
/// outcomes are delegated to [`map_mip_nonsat_result`].
fn map_mip_result_revalidated(
    result: MipResult,
    network: &Network,
    input: &BoundedTensor,
    constraints: &[OutputConstraint],
    num_outputs: usize,
) -> VerificationResult {
    match result {
        MipResult::Sat { input_values, .. } => {
            revalidate_mip_witness(network, input, &input_values, constraints, num_outputs)
        }
        other => map_mip_nonsat_result(other, num_outputs),
    }
}

// #3865: Warm-start vector builder for the sequential PGD→HiGHS path.
#[path = "mip_highs_warm_start.rs"]
pub(super) mod warm_start;

#[path = "mip_highs_intermediate_bounds.rs"]
mod intermediate_bounds;

#[cfg(test)]
#[path = "mip_highs_tests.rs"]
mod tests;
