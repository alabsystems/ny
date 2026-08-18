// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verification dispatch and per-constraint verification functions for ACAS-Xu benchmarks.
//!
//! Contains the core verification logic that routes ACAS-Xu problems to the
//! appropriate backend (sequential Network vs GraphNetwork GPU BaB).

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_onnx::load_onnx;
use ny_onnx::vnnlib::{load_vnnlib, OutputConstraint, VnnLibSpec};
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic, GraphNetwork,
};
use ny_tensor::BoundedTensor;

use crate::commands::beta_crown::branching::{
    parse_branching_heuristic, validate_gpu_bab_branching,
};
use crate::commands::beta_crown::constraint_eval::{
    aggregate_conjunctive, augment_network_with_spec, build_constant_objective,
    build_constant_spec_coeffs, compute_effective_threshold,
};
use crate::commands::beta_crown::constraint_plan::{
    build_constraint_objective, classify_constraints, extract_constant_params, AggregationMode,
    ConstraintObjective,
};
use crate::commands::beta_crown::engine_dispatch::dispatch_graph_constraint;

use super::{AcasxuBenchmarkArgs, AcasxuProblem, GpuBabEngineRuntime};

/// Resolve CLI branching selection, enforcing `--gpu-bab` compatibility rules.
///
/// GPU BaB supports BaBSR (`impact`/`babsr`) and input splitting (`input`).
/// Both are valid for `--gpu-bab`. Unsupported heuristics (width, fsb, sequential)
/// produce errors.
///
/// Part of #1891: InputSplit is now supported for GPU BaB DomainList path.
pub(super) fn resolve_branching_heuristic(
    branching: &str,
    gpu_bab: bool,
) -> Result<BranchingHeuristic> {
    let heuristic = parse_branching_heuristic(branching)?;
    if gpu_bab {
        validate_gpu_bab_branching(&heuristic, branching)?;
    }
    Ok(heuristic)
}

/// Internal verification function.
pub(super) fn run_verification(
    problem: &AcasxuProblem,
    args: &AcasxuBenchmarkArgs,
    gpu_bab_engine: Option<&GpuBabEngineRuntime>,
) -> Result<(BabVerificationStatus, usize, usize)> {
    // Load model
    let onnx_model = load_onnx(&problem.model_path)?;

    // Load property
    let vnnlib = load_vnnlib(&problem.property_path)?;
    let (lower_bounds, upper_bounds) = vnnlib.split_input_bounds_f32();

    // Get input shape from ONNX model (ACAS-Xu uses [1, 1, 1, 5])
    let input_shape: Vec<usize> = onnx_model
        .network
        .inputs
        .first()
        .map(|i| ny_onnx::resolve_dynamic_shape(&i.shape, 1))
        .unwrap_or_else(|| vec![5]);
    let lower = ArrayD::from_shape_vec(IxDyn(&input_shape), lower_bounds)
        .map_err(|e| anyhow::anyhow!("Failed to create lower bounds: {e}"))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&input_shape), upper_bounds)
        .map_err(|e| anyhow::anyhow!("Failed to create upper bounds: {e}"))?;
    let input = BoundedTensor::new(lower, upper)?;

    // Parse branching heuristic and enforce CLI compatibility rules.
    // ReLU splitting requires beta optimization to tighten bounds via Lagrangian
    // relaxation. Without beta_iterations > 0, split constraints have near-zero
    // contribution and bounds may not tighten across splits.
    // Reference: α,β-CROWN paper (Xu et al. 2021), Section 3.2
    let branching_heuristic = resolve_branching_heuristic(&args.branching, args.gpu_bab)?;
    let is_relu_split = !matches!(branching_heuristic, BranchingHeuristic::InputSplit);

    // When using input splitting, auto-enable relaxed clipping (Clip-and-Verify) to match
    // α,β-CROWN's ACAS-Xu configuration. This is critical for achieving comparable pass rates.
    // Reference: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/exp_configs/vnncomp21/acasxu.yaml
    //   bab.clip_n_verify.clip_input_domain.enabled: True
    let is_input_split = matches!(branching_heuristic, BranchingHeuristic::InputSplit);
    let enable_relaxed_clip = args.relaxed_clip || is_input_split;
    let enable_pgd_attack = args.pgd_attack || is_input_split;
    let pgd_restarts = if is_input_split && args.pgd_restarts == 100 {
        // Match α,β-CROWN default: 10000 restarts for ACAS-Xu
        10000
    } else {
        args.pgd_restarts
    };

    // ReLU splitting requires beta optimization to tighten bounds via Lagrangian
    // relaxation. With beta_iterations=0 (default), the beta multipliers stay at
    // 0.0 and split constraints contribute nothing.
    //
    // Also lift beta_max_depth for this benchmark path so optimization is not
    // disabled on deeper ACAS-Xu split domains.
    let beta_iterations = if is_relu_split { 20 } else { 0 };
    let beta_max_depth = if is_relu_split {
        usize::MAX
    } else {
        BetaCrownConfig::default().beta_max_depth
    };

    // Configure verifier using ACAS-Xu preset as base
    let config = BetaCrownConfig {
        max_domains: args.max_domains,
        timeout: std::time::Duration::from_secs(problem.timeout),
        branching_heuristic,
        beta_iterations,
        beta_max_depth,
        enable_proactive_cuts: args.proactive_cuts,
        max_proactive_cuts: args.max_proactive_cuts,
        // Clip-and-Verify: tighten input bounds using CROWN constraints
        enable_relaxed_clip,
        // PGD attack: find counterexamples
        enable_pgd_attack,
        pgd_restarts,
        // lA warm-start: reuse cached linear bounds from parent domain (#1669)
        enable_la_warm_start: !args.no_la_warm_start,
        // Use ACAS-Xu preset for other settings (batch_size: 16384, etc.)
        ..BetaCrownConfig::acas_xu()
    };

    let verifier = match gpu_bab_engine.and_then(|r| r.engine_arc()) {
        Some(engine) => BetaCrownVerifier::new_with_engine(config, engine),
        None => BetaCrownVerifier::new(config),
    };

    // Classify constraints via shared planning module (#1881)
    let classification = classify_constraints(&vnnlib);

    if classification.aggregation == AggregationMode::Disjunctive {
        if args.gpu_bab {
            let graph = onnx_model.to_graph_network()?;
            return verify_disjunctive_gpu_bab(&graph, &input, &vnnlib, &verifier);
        }

        let network = onnx_model.to_propagate_network()?;
        return verify_disjunctive(&network, &input, &vnnlib, &verifier);
    }

    // Route multi-constraint specs (relational, mixed, or multi-constant) through
    // the per-constraint loop which handles all constraint types via
    // build_constraint_objective. Single-constant specs use the faster dedicated path.
    // Matches verify.rs routing: `output_constraints.len() > 1 || has_relational` (#1885).
    let needs_per_constraint = classification.has_relational || vnnlib.output_constraints.len() > 1;

    // Per-constraint decomposition is sound for conjunctive unsafe regions:
    // the BaB verifier tries to prove each constraint Cᵢ is always-violated.
    // If ANY single Cᵢ is always-violated, the conjunction ∧ᵢ Cᵢ can never hold
    // and the network is SAFE (see aggregate_conjunctive in constraint_eval.rs).
    //
    // The multi-objective graph input-split path (verify_graph_input_split_multi_objective_conjunctive)
    // was previously used here but causes performance regressions for ACAS-Xu 4_x:
    // multi-obj explored 100K domains / 0 verified / 3.8s, while per-constraint sequential
    // verified 4_2 in 12.56s. The per-constraint path lets each constraint benefit from
    // focused single-objective alpha-CROWN optimization via input splitting.
    //
    // Ref: #1923 — the multi-objective shortcut was added in W2 1273 (Part of #3218) but
    // regressed ACAS-Xu 4_x which previously verified via per-constraint at eb333968.

    if args.gpu_bab {
        // GPU BaB path: use GraphNetwork + dispatch_graph_constraint.
        // The verifier stores the engine from root construction (#3643).
        let graph = onnx_model.to_graph_network()?;
        if needs_per_constraint {
            verify_relational_gpu_bab(&graph, &input, &vnnlib, &verifier)
        } else {
            verify_constant_gpu_bab(&graph, &input, &vnnlib, &verifier)
        }
    } else {
        // Sequential path: use Network + verify()
        let network = onnx_model.to_propagate_network()?;
        if needs_per_constraint {
            verify_relational(&network, &input, &vnnlib, &verifier)
        } else {
            verify_constant(&network, &input, &vnnlib, &verifier)
        }
    }
}

/// Extract per-clause specs from a VNN-LIB property.
///
/// For disjunctive properties, `output_constraint_clauses` encodes the OR-of-AND
/// structure. For legacy/single-clause specs, fall back to flat `output_constraints`.
fn extract_clause_specs(vnnlib: &VnnLibSpec) -> Vec<VnnLibSpec> {
    let clauses: Vec<Vec<OutputConstraint>> = if vnnlib.output_constraint_clauses.is_empty() {
        vec![vnnlib.output_constraints.clone()]
    } else {
        vnnlib.output_constraint_clauses.clone()
    };

    clauses
        .into_iter()
        .map(|constraints| {
            let mut clause_spec = vnnlib.clone();
            clause_spec.output_constraints = constraints;
            clause_spec.output_constraint_clauses = Vec::new();
            clause_spec.is_disjunction = false;
            clause_spec
        })
        .collect()
}

fn aggregate_disjunctive_clause_results(
    clause_results: &[(BabVerificationStatus, usize, usize)],
) -> (BabVerificationStatus, usize, usize) {
    let mut total_domains = 0usize;
    let mut total_verified = 0usize;
    let mut saw_potential = false;
    let mut unknown_reason: Option<String> = None;

    for (idx, (status, domains, verified)) in clause_results.iter().enumerate() {
        total_domains += domains;
        total_verified += verified;

        match status {
            BabVerificationStatus::Verified => {}
            BabVerificationStatus::Violated {
                counterexample,
                output,
            } => {
                return (
                    BabVerificationStatus::Violated {
                        counterexample: counterexample.clone(),
                        output: output.clone(),
                    },
                    total_domains,
                    total_verified,
                );
            }
            BabVerificationStatus::PotentialViolation { .. } => {
                saw_potential = true;
            }
            BabVerificationStatus::Unknown { reason } => {
                if unknown_reason.is_none() {
                    unknown_reason = Some(format!("Clause {}: {}", idx + 1, reason));
                }
            }
            BabVerificationStatus::Timeout => {
                if unknown_reason.is_none() {
                    unknown_reason = Some(format!("Clause {}: timeout", idx + 1));
                }
            }
        }
    }

    let status = if saw_potential {
        BabVerificationStatus::potential_violation()
    } else if let Some(reason) = unknown_reason {
        BabVerificationStatus::Unknown { reason }
    } else {
        BabVerificationStatus::Verified
    };

    (status, total_domains, total_verified)
}

/// Shared per-constraint iteration with conjunctive aggregation and early exit.
///
/// Iterates over `output_constraints`, builds a `ConstraintObjective` for each,
/// and delegates to `verify_objective` for backend-specific verification.
/// Aggregation via `aggregate_conjunctive`: Verified/Timeout cause early exit.
///
/// Part of #2215: eliminates duplicated iteration in verify_relational/verify_relational_gpu_bab.
fn verify_per_constraint(
    vnnlib: &VnnLibSpec,
    mut verify_objective: impl FnMut(
        &ConstraintObjective,
    ) -> Result<(BabVerificationStatus, usize, usize)>,
) -> Result<(BabVerificationStatus, usize, usize)> {
    let mut results = Vec::new();

    for constraint in &vnnlib.output_constraints {
        let obj = build_constraint_objective(constraint, vnnlib.num_outputs)?;
        let result = verify_objective(&obj)?;
        results.push(result);

        let (status, _, _) = aggregate_conjunctive(&results);
        match status {
            BabVerificationStatus::Verified | BabVerificationStatus::Timeout => {
                return Ok(aggregate_conjunctive(&results));
            }
            _ => {}
        }
    }

    Ok(aggregate_conjunctive(&results))
}

/// Shared disjunctive clause iteration with early exit on violation.
///
/// Extracts per-clause specs, delegates to `verify_clause` for each,
/// and aggregates via `aggregate_disjunctive_clause_results`: concrete
/// violation causes early exit (counterexample found for any clause).
///
/// Part of #2215: eliminates duplicated iteration in verify_disjunctive/verify_disjunctive_gpu_bab.
fn verify_disjunctive_clauses(
    vnnlib: &VnnLibSpec,
    mut verify_clause: impl FnMut(&VnnLibSpec) -> Result<(BabVerificationStatus, usize, usize)>,
) -> Result<(BabVerificationStatus, usize, usize)> {
    let mut clause_results = Vec::new();

    for clause_spec in extract_clause_specs(vnnlib) {
        let clause_result = verify_clause(&clause_spec)?;
        clause_results.push(clause_result);

        let (status, domains, verified) = aggregate_disjunctive_clause_results(&clause_results);
        if matches!(status, BabVerificationStatus::Violated { .. }) {
            return Ok((status, domains, verified));
        }
    }

    Ok(aggregate_disjunctive_clause_results(&clause_results))
}

/// Verify disjunctive properties on sequential networks by checking each clause.
fn verify_disjunctive(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    verifier: &BetaCrownVerifier,
) -> Result<(BabVerificationStatus, usize, usize)> {
    verify_disjunctive_clauses(vnnlib, |clause_spec| {
        let class = classify_constraints(clause_spec);
        let needs_per_constraint = class.has_relational || clause_spec.output_constraints.len() > 1;
        if needs_per_constraint {
            verify_relational(network, input, clause_spec, verifier)
        } else {
            verify_constant(network, input, clause_spec, verifier)
        }
    })
}

/// Verify disjunctive properties on GraphNetwork GPU-BaB path by checking each clause.
fn verify_disjunctive_gpu_bab(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    verifier: &BetaCrownVerifier,
) -> Result<(BabVerificationStatus, usize, usize)> {
    verify_disjunctive_clauses(vnnlib, |clause_spec| {
        let class = classify_constraints(clause_spec);
        let needs_per_constraint = class.has_relational || clause_spec.output_constraints.len() > 1;
        if needs_per_constraint {
            verify_relational_gpu_bab(graph, input, clause_spec, verifier)
        } else {
            verify_constant_gpu_bab(graph, input, clause_spec, verifier)
        }
    })
}

/// Verify property with constant constraints (e.g., Y_0 >= 3.99).
///
/// Uses shared evaluation helpers from constraint_eval.rs (#1881 Step 4).
fn verify_constant(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    verifier: &BetaCrownVerifier,
) -> Result<(BabVerificationStatus, usize, usize)> {
    anyhow::ensure!(
        vnnlib.output_constraints.len() == 1,
        "constant verification requires exactly one scalar output constraint"
    );
    // Extract threshold and output index via shared planning module (#1881)
    let params = extract_constant_params(vnnlib).ok_or_else(|| {
        anyhow::anyhow!(
            "constant verification requires exactly one supported scalar output constraint"
        )
    })?;
    anyhow::ensure!(
        params.output_idx < vnnlib.num_outputs,
        "constant constraint references Y_{} but only {} outputs are declared",
        params.output_idx,
        vnnlib.num_outputs
    );

    // Build augmented network via shared eval helper (#1881 Step 4)
    let spec_coeffs = build_constant_spec_coeffs(&params, vnnlib.num_outputs);
    let augmented = augment_network_with_spec(network, spec_coeffs)?;

    // Compute effective threshold via shared eval helper (#1881 Step 4)
    let effective_threshold = compute_effective_threshold(&params);

    // Preserve the root verifier's stored engine via with_config_from (#3643).
    let adjusted_verifier = verifier.with_config_from(BetaCrownConfig {
        verify_upper_bound: params.verify_upper,
        ..verifier.config.clone()
    });

    let result = adjusted_verifier.verify(&augmented, input, effective_threshold)?;
    Ok((
        result.result,
        result.domains_explored,
        result.domains_verified,
    ))
}

/// Verify property with relational constraints (e.g., Y_1 <= Y_0).
///
/// Uses shared `build_constraint_objective` to handle BOTH relational and constant
/// constraints uniformly (#1881 Step 4, #1888). Iteration and aggregation via
/// `verify_per_constraint` (#2215).
fn verify_relational(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    verifier: &BetaCrownVerifier,
) -> Result<(BabVerificationStatus, usize, usize)> {
    verify_per_constraint(vnnlib, |obj| {
        let augmented = augment_network_with_spec(network, obj.spec_coeffs().to_vec())?;
        let result = verifier.verify(&augmented, input, obj.threshold())?;
        Ok((
            result.result,
            result.domains_explored,
            result.domains_verified,
        ))
    })
}

/// Verify property with constant constraints using GPU BaB (DomainList storage).
///
/// Uses shared evaluation + dispatch helpers for consistent engine wiring (#1881 Step 4).
fn verify_constant_gpu_bab(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    verifier: &BetaCrownVerifier,
) -> Result<(BabVerificationStatus, usize, usize)> {
    anyhow::ensure!(
        vnnlib.output_constraints.len() == 1,
        "constant GPU-BaB verification requires exactly one scalar output constraint"
    );
    // Extract constant params via shared planning module (#1881)
    let params = extract_constant_params(vnnlib).ok_or_else(|| {
        anyhow::anyhow!(
            "constant GPU-BaB verification requires exactly one supported scalar output constraint"
        )
    })?;
    anyhow::ensure!(
        params.output_idx < vnnlib.num_outputs,
        "constant constraint references Y_{} but only {} outputs are declared",
        params.output_idx,
        vnnlib.num_outputs
    );

    // Build objective and threshold via shared eval helpers (#1881 Step 4)
    let objective = build_constant_objective(&params, vnnlib.num_outputs);
    let effective_threshold = compute_effective_threshold(&params);

    // Preserve the root verifier's stored engine via with_config_from (#3643).
    let adjusted_verifier = verifier.with_config_from(BetaCrownConfig {
        verify_upper_bound: params.verify_upper,
        ..verifier.config.clone()
    });

    // Dispatch via shared engine adapter (#1881).
    // Pass None for gemm_engine — the stored engine in adjusted_verifier
    // is resolved internally by dispatch_graph_constraint (#3643).
    let result = dispatch_graph_constraint(
        &adjusted_verifier,
        graph,
        input,
        &objective,
        effective_threshold,
        true, // use_relu_split (GPU BaB implies ReLU splitting)
        true, // gpu_bab
        None, // no precomputed bounds
        None, // use stored engine
        None, // no CLI deadline for benchmarks
    )?;
    Ok((
        result.result,
        result.domains_explored,
        result.domains_verified,
    ))
}

/// Verify property with relational constraints using GPU BaB (DomainList storage).
///
/// Uses shared `build_constraint_objective` for uniform constraint handling
/// and shared engine dispatch (#1881 Step 4, #1888). Iteration and aggregation
/// via `verify_per_constraint` (#2215).
fn verify_relational_gpu_bab(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    verifier: &BetaCrownVerifier,
) -> Result<(BabVerificationStatus, usize, usize)> {
    verify_per_constraint(vnnlib, |obj| {
        let result = dispatch_graph_constraint(
            verifier,
            graph,
            input,
            obj.spec_coeffs(),
            obj.threshold(),
            true, // use_relu_split
            true, // gpu_bab
            None, // no precomputed bounds
            None, // use stored engine
            None, // no CLI deadline for benchmarks
        )?;
        Ok((
            result.result,
            result.domains_explored,
            result.domains_verified,
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use ny_onnx::vnnlib::OutputConstraint;
    use ny_propagate::{layers::LinearLayer, Layer, Network};

    fn make_spec(
        output_constraints: Vec<OutputConstraint>,
        output_constraint_clauses: Vec<Vec<OutputConstraint>>,
        is_disjunction: bool,
    ) -> VnnLibSpec {
        VnnLibSpec {
            num_inputs: 1,
            num_outputs: 3,
            input_bounds: vec![(0.0, 1.0)],
            output_constraints,
            output_constraint_clauses,
            is_disjunction,
            version: None,
            per_clause_input_bounds: Vec::new(),
            declared_input_bounds: Vec::new(),
            dual_network: None,
        }
    }

    fn fixed_scalar_fixture(output: f32) -> (Network, BoundedTensor, BetaCrownVerifier) {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[0.0_f32]]), Some(arr1(&[output])))
                .expect("fixed-output layer should be valid"),
        ));
        let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[0.0_f32]).into_dyn())
            .expect("point input should be valid");
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        (network, input, verifier)
    }

    fn verify_fixed_scalar(output: f32, constraint: OutputConstraint) -> BabVerificationStatus {
        let (network, input, verifier) = fixed_scalar_fixture(output);
        let spec = VnnLibSpec {
            num_outputs: 1,
            ..make_spec(vec![constraint], Vec::new(), false)
        };
        verify_constant(&network, &input, &spec, &verifier)
            .expect("constant verification should run")
            .0
    }

    /// Regression for the historical lower-mode sign inversion. For unsafe
    /// `Y <= 5`, only a strict lower proof `Y > 5` establishes safety.
    #[test]
    fn test_constant_less_eq_uses_original_output_and_direct_threshold() {
        let unsafe_status = verify_fixed_scalar(0.0, OutputConstraint::LessEqConst(0, 5.0));
        assert!(
            !matches!(unsafe_status, BabVerificationStatus::Verified),
            "an actually unsafe fixed output must not be certified safe"
        );

        let safe_status = verify_fixed_scalar(6.0, OutputConstraint::LessEqConst(0, 5.0));
        assert!(matches!(safe_status, BabVerificationStatus::Verified));

        let boundary_status = verify_fixed_scalar(5.0, OutputConstraint::LessEqConst(0, 5.0));
        assert!(
            !matches!(boundary_status, BabVerificationStatus::Verified),
            "the non-strict unsafe boundary must not be certified safe"
        );
    }

    /// The sibling upper-mode obligation remains `upper(Y) < c` for unsafe
    /// `Y >= c`, including strict rejection at the unsafe boundary.
    #[test]
    fn test_constant_greater_eq_uses_original_output_and_direct_threshold() {
        let unsafe_status = verify_fixed_scalar(6.0, OutputConstraint::GreaterEqConst(0, 5.0));
        assert!(
            !matches!(unsafe_status, BabVerificationStatus::Verified),
            "an actually unsafe fixed output must not be certified safe"
        );

        let safe_status = verify_fixed_scalar(4.0, OutputConstraint::GreaterEqConst(0, 5.0));
        assert!(matches!(safe_status, BabVerificationStatus::Verified));

        let boundary_status = verify_fixed_scalar(5.0, OutputConstraint::GreaterEqConst(0, 5.0));
        assert!(
            !matches!(boundary_status, BabVerificationStatus::Verified),
            "the non-strict unsafe boundary must not be certified safe"
        );
    }

    #[test]
    fn test_constant_fast_path_rejects_missing_multiple_and_out_of_range_constraints() {
        let (network, input, verifier) = fixed_scalar_fixture(0.0);

        let missing = VnnLibSpec {
            num_outputs: 1,
            ..make_spec(Vec::new(), Vec::new(), false)
        };
        assert!(verify_constant(&network, &input, &missing, &verifier).is_err());

        let multiple = VnnLibSpec {
            num_outputs: 1,
            ..make_spec(
                vec![
                    OutputConstraint::LessEqConst(0, 1.0),
                    OutputConstraint::GreaterEqConst(0, -1.0),
                ],
                Vec::new(),
                false,
            )
        };
        assert!(verify_constant(&network, &input, &multiple, &verifier).is_err());

        let out_of_range = VnnLibSpec {
            num_outputs: 1,
            ..make_spec(
                vec![OutputConstraint::LessEqConst(1, 1.0)],
                Vec::new(),
                false,
            )
        };
        assert!(verify_constant(&network, &input, &out_of_range, &verifier).is_err());
    }

    #[test]
    fn test_extract_clause_specs_uses_clause_structure_1881() {
        let spec = make_spec(
            vec![
                OutputConstraint::LessEq(0, 1),
                OutputConstraint::LessEq(0, 2),
            ],
            vec![
                vec![OutputConstraint::LessEq(0, 1)],
                vec![
                    OutputConstraint::LessEq(0, 1),
                    OutputConstraint::LessEq(0, 2),
                ],
            ],
            true,
        );

        let clauses = extract_clause_specs(&spec);
        assert_eq!(clauses.len(), 2, "expected two disjunctive clauses");
        assert_eq!(clauses[0].output_constraints.len(), 1);
        assert_eq!(clauses[1].output_constraints.len(), 2);
        assert!(
            clauses
                .iter()
                .all(|clause| !clause.is_disjunction && clause.output_constraint_clauses.is_empty()),
            "clause specs must be normalized to conjunctive form"
        );
    }

    #[test]
    fn test_extract_clause_specs_falls_back_to_flat_constraints_1881() {
        let spec = make_spec(
            vec![OutputConstraint::GreaterEqConst(2, 1.5)],
            Vec::new(),
            true,
        );
        let clauses = extract_clause_specs(&spec);
        assert_eq!(clauses.len(), 1, "flat constraints should form one clause");
        assert_eq!(clauses[0].output_constraints.len(), 1);
        assert!(matches!(
            clauses[0].output_constraints[0],
            OutputConstraint::GreaterEqConst(2, _)
        ));
    }

    #[test]
    fn test_aggregate_disjunctive_clause_results_all_verified_1881() {
        let results = vec![
            (BabVerificationStatus::Verified, 10, 3),
            (BabVerificationStatus::Verified, 20, 7),
        ];
        let (status, domains, verified) = aggregate_disjunctive_clause_results(&results);
        assert!(matches!(status, BabVerificationStatus::Verified));
        assert_eq!(domains, 30);
        assert_eq!(verified, 10);
    }

    #[test]
    fn test_aggregate_disjunctive_clause_results_violation_overrides_1881() {
        let results = vec![
            (BabVerificationStatus::Verified, 10, 3),
            (
                BabVerificationStatus::Violated {
                    counterexample: vec![0.2],
                    output: vec![1.3],
                },
                15,
                0,
            ),
            (BabVerificationStatus::Verified, 100, 100),
        ];
        let (status, domains, verified) = aggregate_disjunctive_clause_results(&results);
        assert!(
            matches!(status, BabVerificationStatus::Violated { .. }),
            "a concrete clause counterexample should mark property violated"
        );
        assert_eq!(
            domains, 25,
            "aggregation should stop at first violated clause"
        );
        assert_eq!(verified, 3);
    }

    #[test]
    fn test_aggregate_disjunctive_clause_results_potential_outranks_unknown_1881() {
        let results = vec![
            (
                BabVerificationStatus::Unknown {
                    reason: "solver gave up".to_string(),
                },
                8,
                0,
            ),
            (BabVerificationStatus::potential_violation(), 12, 0),
        ];
        let (status, domains, verified) = aggregate_disjunctive_clause_results(&results);
        assert!(
            matches!(status, BabVerificationStatus::PotentialViolation { .. }),
            "PotentialViolation should outrank Unknown when no concrete violation exists"
        );
        assert_eq!(domains, 20);
        assert_eq!(verified, 0);
    }

    /// #1885 regression: Multi-constant specs must route through per-constraint loop.
    ///
    /// Before fix: specs with only constant constraints (no relational) always routed
    /// to verify_constant/verify_constant_gpu_bab, which uses extract_constant_params
    /// (first-only via find_map). Multi-constant specs silently lost all but the first.
    ///
    /// After fix: routing uses `has_relational || output_constraints.len() > 1`, so
    /// multi-constant specs go through the per-constraint loop (verify_relational path).
    #[test]
    fn test_multi_constant_routes_to_per_constraint_loop_1885() {
        // Single constant: should NOT need per-constraint loop
        let single_constant = VnnLibSpec {
            num_outputs: 2,
            ..make_spec(
                vec![OutputConstraint::GreaterEqConst(0, 3.99)],
                Vec::new(),
                false,
            )
        };
        let class = classify_constraints(&single_constant);
        let needs_per_constraint =
            class.has_relational || single_constant.output_constraints.len() > 1;
        assert!(
            !needs_per_constraint,
            "single-constant spec should use fast constant path"
        );

        // Multi-constant: MUST route through per-constraint loop
        let multi_constant = VnnLibSpec {
            num_outputs: 2,
            ..make_spec(
                vec![
                    OutputConstraint::GreaterEqConst(0, 3.99),
                    OutputConstraint::GreaterEqConst(1, 3.99),
                ],
                Vec::new(),
                false,
            )
        };
        let class = classify_constraints(&multi_constant);
        let needs_per_constraint =
            class.has_relational || multi_constant.output_constraints.len() > 1;
        assert!(
            needs_per_constraint,
            "multi-constant spec must route through per-constraint loop to avoid dropping constraints"
        );

        // Relational: MUST route through per-constraint loop
        let relational = VnnLibSpec {
            num_outputs: 2,
            ..make_spec(vec![OutputConstraint::LessEq(0, 1)], Vec::new(), false)
        };
        let class = classify_constraints(&relational);
        let needs_per_constraint = class.has_relational || relational.output_constraints.len() > 1;
        assert!(
            needs_per_constraint,
            "relational spec must route through per-constraint loop"
        );

        // Mixed relational + constant: MUST route through per-constraint loop
        let mixed = VnnLibSpec {
            num_outputs: 3,
            ..make_spec(
                vec![
                    OutputConstraint::LessEq(0, 1),
                    OutputConstraint::GreaterEqConst(2, 3.99),
                ],
                Vec::new(),
                false,
            )
        };
        let class = classify_constraints(&mixed);
        let needs_per_constraint = class.has_relational || mixed.output_constraints.len() > 1;
        assert!(
            needs_per_constraint,
            "mixed relational+constant spec must route through per-constraint loop"
        );
    }

    #[test]
    fn test_parse_branching_heuristic() {
        // Test valid heuristics
        assert!(matches!(
            parse_branching_heuristic("width").unwrap(),
            BranchingHeuristic::LargestBoundWidth
        ));
        assert!(matches!(
            parse_branching_heuristic("impact").unwrap(),
            BranchingHeuristic::BoundImpact
        ));
        assert!(matches!(
            parse_branching_heuristic("babsr").unwrap(),
            BranchingHeuristic::BoundImpact
        ));
        assert!(matches!(
            parse_branching_heuristic("fsb").unwrap(),
            BranchingHeuristic::FilteredSmartBranching
        ));
        assert!(matches!(
            parse_branching_heuristic("kfsb").unwrap(),
            BranchingHeuristic::Kfsb
        ));
        assert!(matches!(
            parse_branching_heuristic("sequential").unwrap(),
            BranchingHeuristic::Sequential
        ));
        assert!(matches!(
            parse_branching_heuristic("input").unwrap(),
            BranchingHeuristic::InputSplit
        ));

        // Test invalid heuristic returns error
        assert!(parse_branching_heuristic("unknown").is_err());
        assert!(parse_branching_heuristic("").is_err());
    }

    #[test]
    fn test_resolve_branching_heuristic_gpu_bab_accepts_bound_impact_aliases() {
        let heuristic = resolve_branching_heuristic("impact", true).unwrap();
        assert!(matches!(heuristic, BranchingHeuristic::BoundImpact));

        let heuristic = resolve_branching_heuristic("babsr", true).unwrap();
        assert!(matches!(heuristic, BranchingHeuristic::BoundImpact));
    }

    #[test]
    fn test_resolve_branching_heuristic_gpu_bab_rejects_unsupported_modes() {
        for unsupported in ["width", "fsb", "kfsb", "sequential"] {
            let err = resolve_branching_heuristic(unsupported, true)
                .expect_err("gpu-bab should reject unsupported branching modes in bench_acasxu");
            let msg = err.to_string();
            assert!(
                msg.contains("--gpu-bab supports"),
                "Expected gpu-bab branching compatibility error, got: {msg}"
            );
            assert!(
                msg.contains(unsupported),
                "Expected rejected heuristic '{unsupported}' to appear in error message, got: {msg}"
            );
        }
    }

    /// #1891: `bench -b acasxu --gpu-bab` with default `--branching=input`
    /// now passes through InputSplit directly (no auto-rewrite).
    #[test]
    fn test_resolve_branching_heuristic_gpu_bab_accepts_input_split() {
        // "input" is the clap default for `bench --branching`
        let heuristic = resolve_branching_heuristic("input", true).unwrap();
        assert!(
            matches!(heuristic, BranchingHeuristic::InputSplit),
            "gpu-bab + 'input' should resolve to InputSplit, got: {heuristic:?}"
        );
    }

    /// Without gpu-bab, "input" still resolves to InputSplit normally.
    #[test]
    fn test_resolve_branching_heuristic_no_gpu_bab_input_is_input_split() {
        let heuristic = resolve_branching_heuristic("input", false).unwrap();
        assert!(
            matches!(heuristic, BranchingHeuristic::InputSplit),
            "without gpu-bab, 'input' should be InputSplit, got: {heuristic:?}"
        );
    }

    #[test]
    #[allow(clippy::type_complexity)] // Intentional: compile-time signature assertion
    fn test_gpu_bab_dispatch_functions_require_concrete_engine_signature_2343() {
        // Compile-time guard for #2343: both GPU dispatch helpers must accept a
        // concrete engine reference (not Option), preventing silent `None` wiring.
        // Post-#3643: GPU BaB helpers use the verifier's stored engine
        // instead of accepting a separate `&dyn GemmEngine` parameter.
        let _: fn(
            &GraphNetwork,
            &BoundedTensor,
            &VnnLibSpec,
            &BetaCrownVerifier,
        ) -> Result<(BabVerificationStatus, usize, usize)> = verify_constant_gpu_bab;
        let _: fn(
            &GraphNetwork,
            &BoundedTensor,
            &VnnLibSpec,
            &BetaCrownVerifier,
        ) -> Result<(BabVerificationStatus, usize, usize)> = verify_relational_gpu_bab;
    }
}
