// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph model per-constraint verification.
//!
//! Handles GraphNetwork verification with per-constraint iteration,
//! pre-computed α-CROWN bounds sharing, and multi-objective BaB for disjunctions.

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_core::{f64_to_f32_down, GemmEngine};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier,
    ConjunctiveProofObjectives, GraphNetwork, GraphPrecomputedBounds,
};
use ny_tensor::BoundedTensor;

use super::attack_budget::graph_upfront_pgd_budget;
use super::constraint_iter::{iterate_constraints, ConstraintIterConfig};
use super::disjunctive_pgd::beta_crown_pgd_config;
use super::dispatch_graph_constraint;
use super::graph_pgd::{evaluate_graph, try_graph_pgd_upfront_with_config};
use super::phase_budget::PhaseBudgetLedger;
use super::{build_multi_objectives, classify_constraints, AggregationMode};

/// Build the CLI's narrowly authenticated synthetic-objective plan for the
/// graph input-split lane. Route semantics (graph + conjunction + input
/// splitting) are established by the call site; this helper shares the
/// config/authority predicate with the verifier's typed proof boundary and
/// fails closed on source-property AST or normalized-row drift.
fn exact_conic_proof_objectives(
    config: &BetaCrownConfig,
    vnnlib: &VnnLibSpec,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
) -> Option<ConjunctiveProofObjectives> {
    if !config.input_split_conic_objective_eligible() || !exact_cersyve_conjunctive_spec(vnnlib) {
        return None;
    }
    ConjunctiveProofObjectives::try_exact_two_row_zero_threshold_unit_conic(objectives, thresholds)
}

/// Authenticate the original property AST before any normalized f32 row can
/// acquire derived-objective authority. This deliberately recognizes the
/// parser-real, non-strict Cersyve form only; algebraically similar programmatic
/// or strict properties remain on the historical path.
fn exact_cersyve_conjunctive_spec(vnnlib: &VnnLibSpec) -> bool {
    fn exact_atoms(atoms: &[OutputConstraint]) -> bool {
        matches!(
            atoms,
            [
                OutputConstraint::LessEqConst(0, first),
                OutputConstraint::GreaterEqConst(1, second),
            // IEEE zero signs carry no property meaning. Ordinary equality
            // admits both spellings while rejecting NaN and every nonzero.
            ] if *first == 0.0f64 && *second == 0.0f64
        )
    }

    vnnlib.num_outputs == 2
        && !vnnlib.is_disjunction
        && exact_atoms(&vnnlib.output_constraints)
        && matches!(
            vnnlib.output_constraint_clauses.as_slice(),
            [clause] if exact_atoms(clause)
        )
        && vnnlib.per_clause_input_bounds.is_empty()
        && vnnlib.dual_network.is_none()
}

fn admits_graph_upfront_pgd(
    run_upfront_pgd: bool,
    is_disjunction: bool,
    clause_count: usize,
) -> bool {
    run_upfront_pgd && (!is_disjunction || clause_count <= 1)
}

/// Verify graph model with relational constraints.
// Justification: Graph verification requires model, bounds, constraints, config,
// verifier, BaB flags, attack parameters, and engine — all independent inputs.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_graph_relational(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    use_relu_split: bool,
    gpu_bab: bool,
    run_upfront_pgd: bool,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    // #attack-steering-conjunctive: falsification-only accelerator channel.
    // Reaches ONLY the upfront graph PGD below; every bound/BaB consumer in
    // this function keeps `gemm_engine` (the quarantined proof handle).
    attack_engine_source: crate::commands::beta_crown::attack_arming::AttackEngineSource<'_>,
    json: bool,
    ledger: &PhaseBudgetLedger,
) -> Result<BetaCrownResult> {
    // Classify constraints via shared planning module (#1881)
    let classification = classify_constraints(vnnlib);
    let total_constraint_count = vnnlib.output_constraints.len();
    let is_disjunction = classification.aggregation == AggregationMode::Disjunctive;
    // The caller has already applied static MIP eligibility to its
    // authoritative ledger. Keep that start time and reservation policy intact.
    ledger.emit_telemetry("graph-enter");
    let start_time = std::time::Instant::now();

    // Early PGD attack for graph networks before expensive α-CROWN computation.
    // For Conv-heavy models (soundnessbench, malbeware conv), α-CROWN bounds
    // take 30-60s. Random sampling + SPSA PGD can find violations in <1s.
    //
    // Single-clause disjunctions (output_constraint_clauses.len() <= 1) are
    // semantically conjunctive — all constraints in the single clause must hold
    // simultaneously. PGD on the conjunctive interpretation is correct. (#3309)
    //
    // Reserve (1 - upfront_pgd_fraction) of timeout for CROWN + BaB (#3781).
    // Graph PGD can be expensive on large models.
    if admits_graph_upfront_pgd(
        run_upfront_pgd,
        is_disjunction,
        vnnlib.output_constraint_clauses.len(),
    ) {
        let (pgd_restarts, pgd_steps) = graph_upfront_pgd_budget(config);
        let pgd_deadline = ledger.upfront_pgd_deadline();
        // #attack-steering-conjunctive: take the falsification accelerator (a
        // non-blocking take: `None` while arming / when disarmed) and prefer it
        // over the proof handle. `b030e2a8` restored this channel for the
        // DISJUNCTIVE lane only; the conjunctive graph lane below kept reading
        // the proof `gemm_engine`, which `1ede1d30` hard-`None`d — so on every
        // host the batched exact-VJP wave (`graph_pgd_vjp_batched.rs`, gated on
        // `as_gpu_crown_backward`) statically declined and this lane fell to the
        // sequential exact-gradient loop. MEASURED on soundnessbench model_0 at
        // the official 150 s budget: 8 sequential restart-steps in the whole
        // 121 s upfront slice, versus the batched wave's ~690 steps × 52 lanes.
        // Verdict-neutral: gradients only choose WHERE to look, and every
        // candidate still passes `revalidate_graph_counterexample`.
        //
        // #attack-steering-arming-race: this lane takes the engine ONCE, and
        // its slice is the bulk of the instance, so a not-yet-armed engine
        // would be lost for the whole slice. Wait at most 1% of this lane's
        // OWN slice (hard cap 500 ms) for arming to settle — on a near-wall
        // row whose attack window is milliseconds the wait is proportionally
        // invisible, which is the property A6 protects.
        let arming_grace = pgd_deadline
            .map(|d| d.saturating_duration_since(std::time::Instant::now()) / 100)
            .unwrap_or(std::time::Duration::from_millis(500))
            .min(std::time::Duration::from_millis(500));
        let attack_take = attack_engine_source.take_within(arming_grace);
        let attack_engine = attack_take
            .as_ref()
            .map(|taken| taken.as_gemm())
            .or(gemm_engine);
        if let Some((counterexample, output)) = try_graph_pgd_upfront_with_config(
            graph,
            input,
            vnnlib,
            beta_crown_pgd_config(config, pgd_restarts, pgd_steps, pgd_deadline),
            attack_engine,
            json,
        )? {
            return Ok(BetaCrownResult {
                result: BabVerificationStatus::Violated {
                    counterexample: counterexample.iter().copied().collect(),
                    output: output.iter().copied().collect(),
                },
                domains_explored: 0,
                domains_verified: 0,
                cuts_generated: 0,
                max_depth_reached: 0,
                time_elapsed: start_time.elapsed(),
                output_bounds: None,
            });
        }
    }

    // Per-constraint timeout derived from ledger remaining time (#2206 Packet B).
    // Uses remaining wall-clock after PGD instead of the original raw timeout.
    // - Conjunctive: full remaining per constraint (early exit on first success).
    // - Disjunctive: splits remaining evenly, floored to min_clause_secs.
    let per_constraint_timeout = ledger
        .per_clause_timeout(total_constraint_count, is_disjunction)
        .unwrap_or(std::time::Duration::from_secs(timeout));

    // Certified sparse-input double-double zonotope (#dd-zonotope, default-ON;
    // `NY_DD_ZONOTOPE=0` is the kill switch). Its structural detector still
    // declines ordinary small-input models before doing proof work.
    //
    // This sits AHEAD of both the multi-objective and per-constraint branches
    // because `vggnet16_2022` specs carry exactly ONE output constraint, so the
    // multi-objective engine hook in
    // `beta_crown::engine::graph::multi_objective::root` is never reached for
    // them; and because the existing root pass on that category consumes the
    // whole budget, so an intersect placed after it would never run either.
    //
    // The pass is a pure ADDITION: it can only return "every objective is
    // certified", in which case the property is safe. Any refusal (gate off,
    // detector declined, unsupported op, cap exceeded, deadline, or the
    // self-policing precision gate) falls straight through to the code below
    // unchanged, so gate-off is byte-identical. See
    // `ny_propagate::dd_zonotope` for the soundness contract.
    if ny_propagate::dd_zonotope::dd_zonotope_enabled() {
        if let Some(result) =
            try_dd_zonotope_root(graph, input, vnnlib, config, ledger, start_time, json)?
        {
            return Ok(result);
        }
    }

    // Multi-objective BaB for ≥2 constraints: conjunctive uses joint BaB (any_verified
    // semantics, strictly more powerful than per-constraint decomposition), disjunctive
    // uses all_verified semantics. Input splitting falls back to per-constraint for
    // disjunctive. See designs/2026-03-05-joint-conjunctive-bab.md.
    if total_constraint_count >= 2 && (use_relu_split || !is_disjunction) {
        let conjunctive = !is_disjunction;
        let (objectives, thresholds) = build_multi_objectives(vnnlib)?;

        if !json {
            let split_type = if use_relu_split {
                "ReLU splitting"
            } else {
                "input splitting"
            };
            let mode = if conjunctive {
                "conjunction (AND)"
            } else {
                "disjunction (OR)"
            };
            let safe_desc = if conjunctive {
                "ANY single constraint provably violated"
            } else {
                "ALL constraints provably violated"
            };
            println!("\nRunning multi-objective Graph β-CROWN ({split_type}) for {mode}...");
            println!(
                "Verifying {} constraints simultaneously (shared computation)",
                objectives.len()
            );
            println!("SAFE requires: {safe_desc}");
        }

        let multi_verifier = verifier.with_config_from(config.clone());
        ledger.emit_telemetry("graph-multi-objective-handoff");
        let bab_deadline = ledger.bab_deadline();
        let result = if conjunctive {
            if use_relu_split {
                multi_verifier.verify_graph_relu_split_multi_objective_conjunctive_with_engine(
                    graph,
                    input,
                    &objectives,
                    &thresholds,
                    gemm_engine,
                    bab_deadline,
                )?
            } else if let Some(proof_objectives) =
                exact_conic_proof_objectives(config, vnnlib, &objectives, &thresholds)
            {
                tracing::info!(
                    treatment = "input_split_selective_direct_and_affine_conic_closure",
                    configured = true,
                    route_eligible = true,
                    source_ast_mutated = false,
                    materialized_derived_rows = proof_objectives.len() - objectives.len(),
                    source_crown_rows = objectives.len(),
                    available_derived_rows = proof_objectives.len() - objectives.len(),
                    direct_strategy = "selective_root_and_ranked_microbatches",
                    original_rows = objectives.len(),
                    proof_rows = proof_objectives.len(),
                    provenance = ?proof_objectives.provenance(),
                    "admitted exact authenticated affine conic closure"
                );
                multi_verifier.verify_graph_input_split_conjunctive_proof_objectives(
                    graph,
                    input,
                    &proof_objectives,
                    gemm_engine,
                    bab_deadline,
                )?
            } else {
                multi_verifier.verify_graph_input_split_multi_objective_conjunctive(
                    graph,
                    input,
                    &objectives,
                    &thresholds,
                    gemm_engine,
                    bab_deadline,
                )?
            }
        } else {
            multi_verifier.verify_graph_relu_split_multi_objective_with_engine(
                graph,
                input,
                &objectives,
                &thresholds,
                gemm_engine,
                bab_deadline,
            )?
        };

        if !json {
            println!("\n  Multi-objective result: {:?}", result.result);
            println!("    Domains explored: {}", result.domains_explored);
            println!("    Domains verified: {}", result.domains_verified);
            println!("    Max depth: {}", result.max_depth_reached);
            println!("    Time: {:.2}s", result.time_elapsed.as_secs_f64());
        }

        return Ok(result);
    }

    // Per-constraint verification (non-disjunction or input splitting)
    if !json {
        let split_type = if use_relu_split {
            "ReLU splitting"
        } else {
            "input splitting"
        };
        println!(
            "\nRunning Graph β-CROWN ({}) with constraints...",
            split_type
        );
        if is_disjunction {
            println!("Disjunctive property: SAFE if ALL constraints are provably violated.");
            println!(
                "Timeout budget: {:.1}s per constraint ({} constraints)",
                per_constraint_timeout.as_secs_f64(),
                total_constraint_count
            );
        } else {
            println!("Conjunctive property: SAFE if ANY constraint is provably violated.");
            println!(
                "Early-exit strategy: full timeout ({:.1}s) per constraint, stop on first success ({} constraints)",
                per_constraint_timeout.as_secs_f64(),
                total_constraint_count
            );
        }
    }

    // Use remaining time from ledger rather than the original raw timeout,
    // so earlier phases (PGD, multi-objective BaB) reduce the per-constraint budget.
    let overall_timeout = ledger
        .remaining()
        .unwrap_or(std::time::Duration::from_secs(timeout));

    // Pre-compute α-CROWN bounds once for all constraints (major optimization)
    // This avoids re-computing bounds for each constraint, providing ~Nx speedup
    let precomputed_bounds = if use_relu_split && total_constraint_count > 1 {
        if !json {
            println!(
                "\n  Pre-computing α-CROWN bounds (shared across {} constraints)...",
                total_constraint_count
            );
        }
        let bounds_start = std::time::Instant::now();
        let bounds_verifier = verifier.with_config_from(config.clone());
        // Thread deadline from ledger so α-CROWN bails early if timeout budget
        // exhausted (#2698). Uses ledger.overall_deadline() instead of manually
        // computing overall_start + overall_timeout.
        let deadline = ledger.overall_deadline();
        let (node_bounds, output_bounds) =
            bounds_verifier.compute_initial_graph_bounds(graph, input, deadline)?;
        if !json {
            println!(
                "    Bounds computed in {:.2}s",
                bounds_start.elapsed().as_secs_f64()
            );
        }
        // Stash for a potential Graph-MIP escalation at BaB fall-through
        // (increment 5): the escalation reuses THIS per-property α-CROWN map
        // instead of recomputing it. The bounded stash is disabled when its
        // whole-network Graph-MIP consumer is explicitly off or this category
        // requests no MIP reservation.
        #[cfg(feature = "mip")]
        super::super::graph_mip_escalate::stash_graph_bounds_for_mip(
            graph,
            input,
            &config.phase_budget,
            &node_bounds,
        );
        Some((node_bounds, output_bounds))
    } else {
        None
    };

    let iter_config = ConstraintIterConfig {
        aggregation: classification.aggregation,
        overall_timeout,
        per_constraint_timeout,
        min_timeout_ms: 100,
        total_constraint_count,
        num_outputs: vnnlib.num_outputs,
        base_config: config.clone(),
        parent_verifier: Some(verifier),
        engine: verifier.engine_arc(),
        json,
    };

    // Cross-validation closure: evaluate original (un-augmented) graph network at a
    // concrete counterexample input. Part of #3209.
    let eval_original = |cx_input: &[f32]| -> Result<ArrayD<f32>> {
        let point = ArrayD::from_shape_vec(IxDyn(input.lower().shape()), cx_input.to_vec())?;
        evaluate_graph(graph, &point, gemm_engine)
    };

    let bab_deadline = ledger.bab_deadline();
    iterate_constraints(
        &vnnlib.output_constraints,
        &iter_config,
        |dispatch| {
            let bounds_ref = precomputed_bounds
                .as_ref()
                .map(|(nb, ob)| GraphPrecomputedBounds::new(nb, ob));
            dispatch_graph_constraint(
                dispatch.verifier,
                graph,
                input,
                dispatch.spec_coeffs,
                dispatch.threshold,
                use_relu_split,
                gpu_bab,
                bounds_ref.as_ref(),
                gemm_engine,
                bab_deadline,
            )
        },
        Some(&eval_original),
    )
}

/// Try to certify EVERY output constraint with the double-double zonotope
/// (`#dd-zonotope`). `Ok(None)` on any refusal — never a degraded verdict.
///
/// A `Verified` return here means the certified margin lower bound cleared the
/// threshold on every objective. Two independent enclosures are NOT combined:
/// this is the zonotope's own certificate, produced by a pass whose rounding
/// channel is computed rather than assumed and which refuses when that channel
/// is not far below the margin.
fn try_dd_zonotope_root(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    config: &BetaCrownConfig,
    ledger: &PhaseBudgetLedger,
    start_time: std::time::Instant,
    json: bool,
) -> Result<Option<BetaCrownResult>> {
    use ny_propagate::dd_zonotope::{dd_zonotope_margins, DdZonoConfig, DdZonoPlan};

    // Admission caps only (#metaroom-ddzono): the category preset may resize
    // the detector's blast-radius/resource caps (no preset section leaves them
    // byte-identical); explicitly set env knobs keep precedence, and the
    // soundness gates are not preset-reachable.
    let cfg = DdZonoConfig::from_env().with_admission_overrides(
        config.dd_zonotope_min_input_numel,
        config.dd_zonotope_max_k,
        config.dd_zonotope_max_generators,
        config.dd_zonotope_collect_interm,
    );
    let Some(plan) = DdZonoPlan::detect(graph, input, &cfg) else {
        if !json {
            println!("[dd-zonotope] detector declined; continuing with the standard pipeline");
        }
        return Ok(None);
    };
    let (objectives, thresholds) = build_multi_objectives(vnnlib)?;
    if objectives.is_empty() || objectives.len() != thresholds.len() {
        return Ok(None);
    }

    // Bounded slice of the remaining budget, so a refusal never costs the
    // whole timeout.
    let now = std::time::Instant::now();
    let cap = std::time::Duration::from_secs(
        std::env::var("NY_DD_ZONOTOPE_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(900),
    );
    let deadline = ledger.bab_deadline().map_or(now + cap, |d| {
        now + cap.min(d.saturating_duration_since(now).mul_f32(0.6))
    });
    if !json {
        println!(
            "[dd-zonotope] admitted: k={} input={} objectives={} budget={:.1}s",
            plan.k(),
            input.len(),
            objectives.len(),
            deadline.saturating_duration_since(now).as_secs_f32()
        );
    }

    let margin = match dd_zonotope_margins(graph, input, &objectives, &plan, &cfg, Some(deadline)) {
        Ok(Some(m)) => m,
        Ok(None) => {
            if !json {
                println!("[dd-zonotope] refused (fail-closed); bounds unchanged");
            }
            return Ok(None);
        }
        Err(e) => {
            if !json {
                println!("[dd-zonotope] error ({e}); bounds unchanged");
            }
            return Ok(None);
        }
    };
    if !json {
        for (i, ((&l, &u), (&hw, &rw))) in margin
            .lower
            .iter()
            .zip(margin.upper.iter())
            .zip(
                margin
                    .rounding_half_width
                    .iter()
                    .zip(margin.relax_half_width.iter()),
            )
            .enumerate()
        {
            println!(
                "[dd-zonotope] obj[{i}] certified=[{l:.9}, {u:.9}] relax_hw={rw:.4e} rounding_hw={hw:.4e} \
                 lower@{:.1}x_rounding={:.9} threshold={}",
                cfg.safety_factor,
                margin.lower_with_safety(i, cfg.safety_factor),
                thresholds[i]
            );
        }
        println!(
            "[dd-zonotope] gens={} wall={:.1}s",
            margin.n_generators,
            margin.wall.as_secs_f32()
        );
    }

    // SELF-POLICING PRECISION GATE: the certified ROUNDING half-width is
    // computed, not assumed. The ~2^66 amplification behind this method was
    // MEASURED on vgg16-7 only; a deeper or larger-weight network could exceed
    // double-double, and this turns that from an unsound assumption into an
    // ordinary decline.
    if !margin.precision_ok(cfg.precision_ratio) {
        if !json {
            println!(
                "[dd-zonotope] PRECISION GATE refused: the certified rounding half-width is \
                 not below {:.1e} x |margin|; bounds unchanged",
                cfg.precision_ratio
            );
        }
        return Ok(None);
    }

    // VERDICT with the explicit safety factor on the certified rounding
    // channel: the property must still hold when that channel is inflated by
    // `cfg.safety_factor` (default 2x). The bit-classified `f64 -> f32`
    // narrowing rounds OUTWARD, so neither the hardware rounding mode nor
    // FTZ/DAZ can round a certified lower bound UP across the threshold.
    // `objectives.len() ==
    // thresholds.len()` was checked on entry, and `evaluate_objectives` emits
    // one margin per objective, so the three lengths agree here.
    let all_verified = margin.lower.len() == thresholds.len()
        && thresholds.iter().enumerate().all(|(i, &t)| {
            let lo = margin.lower_with_safety(i, cfg.safety_factor);
            lo.is_finite() && f64_to_f32_down(lo) > t
        });
    if !all_verified {
        if !json {
            println!("[dd-zonotope] certified margin does not clear every threshold; continuing");
        }
        return Ok(None);
    }

    let output_bounds = build_dd_output_bounds(&margin);
    if !json {
        println!(
            "[dd-zonotope] all {} objective(s) CERTIFIED at the root — property safe",
            objectives.len()
        );
    }
    Ok(Some(BetaCrownResult {
        result: BabVerificationStatus::Verified,
        domains_explored: 1,
        domains_verified: 1,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: start_time.elapsed(),
        output_bounds,
    }))
}

fn build_dd_output_bounds(
    margin: &ny_propagate::dd_zonotope::DdZonoMargin,
) -> Option<BoundedTensor> {
    if margin.output_lower.is_empty() || margin.output_shape.is_empty() {
        return None;
    }
    let shape = IxDyn(&margin.output_shape);
    let lower = ArrayD::from_shape_vec(shape.clone(), margin.output_lower.clone()).ok()?;
    let upper = ArrayD::from_shape_vec(shape, margin.output_upper.clone()).ok()?;
    BoundedTensor::new(lower, upper).ok()
}

#[cfg(test)]
mod conic_proof_objective_tests {
    use super::*;
    use ny_onnx::vnnlib::{
        parse_vnnlib, DualNetworkProperty, DualNetworkSpec, DualNetworkValidation,
    };
    use ny_propagate::VerificationArtifactAuthority;

    fn exact_rows() -> (Vec<Vec<f32>>, Vec<f32>) {
        (vec![vec![1.0, 0.0], vec![0.0, -1.0]], vec![0.0, -0.0])
    }

    #[test]
    fn zero_upfront_slice_declines_before_exact_vjp_plan_construction() {
        assert!(!admits_graph_upfront_pgd(false, false, 1));
        assert!(!admits_graph_upfront_pgd(false, true, 1));
        assert!(!admits_graph_upfront_pgd(false, true, 2));
        assert!(admits_graph_upfront_pgd(true, false, 1));
    }

    fn exact_spec() -> VnnLibSpec {
        let atoms = vec![
            OutputConstraint::LessEqConst(0, 0.0),
            OutputConstraint::GreaterEqConst(1, 0.0),
        ];
        let mut spec = VnnLibSpec::new();
        spec.num_outputs = 2;
        spec.output_constraints = atoms.clone();
        spec.output_constraint_clauses = vec![atoms];
        spec
    }

    fn placeholder_dual_network() -> DualNetworkSpec {
        DualNetworkSpec {
            networks: Vec::new(),
            property: DualNetworkProperty::EpsilonEquivalence { epsilon: 0.0 },
            shared_input_coupling: false,
            f_input_bounds: Vec::new(),
            g_input_bounds: Vec::new(),
            validation: DualNetworkValidation {
                input_equalities: Vec::new(),
                f_input_ge_g_input: Vec::new(),
                g_input_ge_f_input: Vec::new(),
                isomorphic_output_safe_complement: false,
                monotonic_output_relation_count: 0,
                unsupported_output_relation: false,
                isomorphic_output_atoms: Vec::new(),
                isomorphic_output_is_conjunction: true,
            },
            formula_dnf: None,
        }
    }

    #[test]
    fn route_requires_both_typed_gate_and_verdict_only_authority() {
        let (objectives, thresholds) = exact_rows();
        let vnnlib = exact_spec();
        let mut config = BetaCrownConfig::default();
        assert!(exact_conic_proof_objectives(&config, &vnnlib, &objectives, &thresholds).is_none());

        config.input_split_conic_objective = true;
        assert_eq!(
            config.verification_artifact_authority,
            VerificationArtifactAuthority::CertificateExport
        );
        assert!(exact_conic_proof_objectives(&config, &vnnlib, &objectives, &thresholds).is_none());

        config.verification_artifact_authority = VerificationArtifactAuthority::VerdictOnly;
        let plan = exact_conic_proof_objectives(&config, &vnnlib, &objectives, &thresholds)
            .expect("verdict-only exact route should admit the authenticated plan");
        assert_eq!(plan.len(), 3);
    }

    #[test]
    fn parser_real_cersyve_shape_survives_planning_and_authentication() {
        let vnnlib = parse_vnnlib(
            r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (and (<= Y_0 0.0) (>= Y_1 0.0)))
"#,
        )
        .expect("parser-real Cersyve-shaped property");
        assert!(exact_cersyve_conjunctive_spec(&vnnlib));

        let (objectives, thresholds) =
            build_multi_objectives(&vnnlib).expect("normalized objectives build");
        assert_eq!(objectives, vec![vec![1.0, 0.0], vec![0.0, -1.0]]);
        // Both thresholds are ZERO. The SIGN is deliberately not pinned.
        //
        // `exact_cersyve_conjunctive_spec` states the principle directly --
        // "IEEE zero signs carry no property meaning" -- so pinning a bit
        // pattern here contradicts the gate this test exercises. It also does
        // not survive contact: the parser emits `-0.0` for both atoms, and
        // whether each survives normalization as `-0.0` or flips to `+0.0`
        // depends on how a row is built, which the conic work has now changed
        // twice. Both spellings denote the same real bound and feed the same
        // comparison, and `== 0.0` still rejects NaN and every nonzero.
        assert_eq!(thresholds[0], 0.0f32);
        assert_eq!(thresholds[1], 0.0f32);

        let config = BetaCrownConfig {
            verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
            input_split_conic_objective: true,
            ..Default::default()
        };
        assert!(exact_conic_proof_objectives(&config, &vnnlib, &objectives, &thresholds).is_some());
    }

    #[test]
    fn property_authentication_rejects_strict_and_nonzero_lookalikes() {
        for atoms in [
            vec![
                OutputConstraint::LessThanConst(0, 0.0),
                OutputConstraint::GreaterEqConst(1, 0.0),
            ],
            vec![
                OutputConstraint::LessEqConst(0, 0.0),
                OutputConstraint::GreaterThanConst(1, 0.0),
            ],
            vec![
                OutputConstraint::LessEqConst(0, 1.0),
                OutputConstraint::GreaterEqConst(1, 0.0),
            ],
        ] {
            let mut spec = exact_spec();
            spec.output_constraints = atoms.clone();
            spec.output_constraint_clauses = vec![atoms];
            assert!(!exact_cersyve_conjunctive_spec(&spec));
        }
    }

    #[test]
    fn property_authentication_accepts_every_signed_zero_spelling() {
        for (first, second) in [(0.0, 0.0), (0.0, -0.0), (-0.0, 0.0), (-0.0, -0.0)] {
            let atoms = vec![
                OutputConstraint::LessEqConst(0, first),
                OutputConstraint::GreaterEqConst(1, second),
            ];
            let mut spec = exact_spec();
            spec.output_constraints = atoms.clone();
            spec.output_constraint_clauses = vec![atoms];
            assert!(exact_cersyve_conjunctive_spec(&spec));
        }
    }

    #[test]
    fn property_authentication_rejects_clause_and_route_semantic_drift() {
        let mut missing_clause = exact_spec();
        missing_clause.output_constraint_clauses.clear();
        assert!(!exact_cersyve_conjunctive_spec(&missing_clause));

        let mut mismatched_clause = exact_spec();
        mismatched_clause.output_constraint_clauses[0].swap(0, 1);
        assert!(!exact_cersyve_conjunctive_spec(&mismatched_clause));

        let mut extra_clause = exact_spec();
        extra_clause
            .output_constraint_clauses
            .push(extra_clause.output_constraints.clone());
        assert!(!exact_cersyve_conjunctive_spec(&extra_clause));

        let mut per_clause_box = exact_spec();
        per_clause_box.per_clause_input_bounds = vec![Default::default()];
        assert!(!exact_cersyve_conjunctive_spec(&per_clause_box));

        let mut dual_network = exact_spec();
        dual_network.dual_network = Some(placeholder_dual_network());
        assert!(!exact_cersyve_conjunctive_spec(&dual_network));

        let mut disjunction = exact_spec();
        disjunction.is_disjunction = true;
        assert!(!exact_cersyve_conjunctive_spec(&disjunction));

        let mut extra_output = exact_spec();
        extra_output.num_outputs = 3;
        assert!(!exact_cersyve_conjunctive_spec(&extra_output));
    }
}
