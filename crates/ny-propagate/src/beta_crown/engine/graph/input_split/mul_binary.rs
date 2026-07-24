// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! MulBinary SPSA alpha optimization for input-split BaB.
//!
//! Initializes and optimizes MulBinary alphas (McCormick facet interpolation
//! parameters) using SPSA gradient estimation. The optimized alphas are
//! frozen at the root domain and reused for all per-domain CROWN passes.
//!
//! Part of #3439 Phase 4.

use std::collections::HashMap;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info};

use crate::layers::Layer;
use crate::GraphNetwork;

use super::shared::{
    graph_crown_error_should_fallback, graph_spec_crown_with_mul_binary_and_truncation,
};

pub(crate) fn graph_has_mul_binary(graph: &GraphNetwork) -> bool {
    graph
        .nodes
        .values()
        .any(|node| matches!(&node.layer, Layer::MulBinary(_)))
}

pub(crate) fn maybe_optimize_mul_binary_alphas(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    log_prefix: &str,
) -> Result<Option<HashMap<String, Array2<f32>>>> {
    if !graph_has_mul_binary(graph) {
        return Ok(None);
    }

    if deadline.is_some_and(|d| Instant::now() >= d) {
        debug!(
            "{}: skipping MulBinary alpha optimization because the verifier deadline already expired",
            log_prefix
        );
        return Ok(None);
    }

    let ibp_bounds = match graph
        .collect_crown_ibp_bounds_dag_with_deadline_and_engine(input, deadline, engine)
    {
        Ok(bounds) => bounds,
        Err(err) if graph_crown_error_should_fallback(&err) => {
            debug!(
                "{}: MulBinary CROWN-IBP prepass failed with {}, using default relaxations",
                log_prefix, err
            );
            return Ok(None);
        }
        Err(err) => return Err(err),
    };

    let mut alphas = init_mul_binary_alphas(graph, &ibp_bounds);
    if alphas.is_empty() {
        return Ok(None);
    }

    let spsa_iterations = 20;
    let spsa_lr = 0.1;
    info!(
        "{}: optimizing {} MulBinary alpha sets with {} SPSA iterations...",
        log_prefix,
        alphas.len(),
        spsa_iterations
    );
    let optimized_any = optimize_mul_binary_alphas_spsa(
        graph,
        input,
        spec_matrix,
        &mut alphas,
        engine,
        deadline,
        crown_backward_layers,
        spsa_iterations,
        spsa_lr,
    )?;
    if deadline.is_some_and(|d| Instant::now() >= d) && !optimized_any {
        debug!(
            "{}: deadline expired before any MulBinary SPSA step completed, using default relaxations",
            log_prefix
        );
        return Ok(None);
    }
    debug!("{}: MulBinary alpha optimization complete", log_prefix);
    Ok(Some(alphas))
}

/// Initialize MulBinary alpha parameters for all MulBinary nodes in the graph.
///
/// Returns a map from node name to `[2, n]` array where row 0 = r_l (lower facet
/// interpolation) and row 1 = r_u (upper facet interpolation). Initialized to 0.5
/// (equivalent to default McCormick Middle mode).
///
/// Part of #3439 Phase 4.
fn init_mul_binary_alphas(
    graph: &GraphNetwork,
    node_bounds: &HashMap<String, BoundedTensor>,
) -> HashMap<String, Array2<f32>> {
    let mut alphas = HashMap::new();
    for (name, node) in &graph.nodes {
        if let Layer::MulBinary(_) = &node.layer {
            if let Some(bounds) = node_bounds.get(name) {
                let n = bounds.lower().len();
                alphas.insert(name.clone(), Array2::from_elem((2, n), 0.5));
            }
        }
    }
    alphas
}

/// Optimize MulBinary alphas using targeted SPSA gradient estimation.
///
/// Perturbs MulBinary alphas with Bernoulli ±1 perturbations and uses
/// finite-difference gradient estimation to select tighter McCormick facets.
/// ReLU relaxation uses default slopes (not optimized here — that's the
/// alpha-CROWN loop's job).
///
/// Reference: Same SPSA approach as propagate_dag.rs MulBinary supplement
/// (lines 1002-1120), adapted for spec-guided CROWN backward.
/// Part of #3439 Phase 4.
#[allow(clippy::too_many_arguments)]
fn optimize_mul_binary_alphas_spsa(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    alphas: &mut HashMap<String, Array2<f32>>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    iterations: usize,
    lr: f32,
) -> Result<bool> {
    use rand::RngExt;

    if alphas.is_empty() || iterations == 0 {
        return Ok(false);
    }

    let eps = 1e-3_f32;
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let adam_eps = 1e-8_f32;
    let mut optimized_any = false;

    // Adam state per MulBinary node.
    let mut adam_m: HashMap<String, Array2<f32>> = alphas
        .iter()
        .map(|(k, v)| (k.clone(), Array2::zeros(v.raw_dim())))
        .collect();
    let mut adam_v: HashMap<String, Array2<f32>> = alphas
        .iter()
        .map(|(k, v)| (k.clone(), Array2::zeros(v.raw_dim())))
        .collect();

    let mut rng = crate::random::rng();

    for iter in 0..iterations {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            debug!(
                "MulBinary SPSA: deadline exceeded before iteration {}/{}, returning current alphas",
                iter + 1,
                iterations
            );
            return Ok(optimized_any);
        }

        // Generate Bernoulli ±1 perturbations for all MulBinary alphas.
        //
        // Determinism (task #36): draw the perturbations in SORTED-key order so the
        // seeded RNG is consumed in a process-order-INDEPENDENT sequence. Iterating
        // `alphas` (a `HashMap`) directly consumed the RNG in the map's random
        // per-process iteration order, so with more than one MulBinary alpha set the
        // SAME seeded draws were assigned to DIFFERENT nodes from one run to the
        // next. That produced different (all sound) optimized alphas, hence a
        // different input-split BaB tree and a run-dependent verdict on razor-thin
        // disjunctive specs (lsnc `quadrotor2d_state_0`). The ReLU α SPSA already
        // avoids this by keying its perturbations through a `BTreeMap`; this mirrors
        // that discipline. Order only affects WHICH sound relaxation is picked, so it
        // never weakens a bound.
        let mut pert_names: Vec<&String> = alphas.keys().collect();
        pert_names.sort_unstable();
        let perturbations: HashMap<String, Array2<f32>> = pert_names
            .into_iter()
            .map(|name| {
                let alpha = &alphas[name];
                let pert = Array2::from_shape_fn(alpha.raw_dim(), |_| {
                    if rng.random_bool(0.5) {
                        1.0_f32
                    } else {
                        -1.0_f32
                    }
                });
                (name.clone(), pert)
            })
            .collect();

        // +eps perturbation
        let mut alpha_plus: HashMap<String, Array2<f32>> = alphas.clone();
        for (name, pert) in &perturbations {
            if let Some(a) = alpha_plus.get_mut(name) {
                a.zip_mut_with(pert, |a_val, &p| {
                    *a_val = (*a_val + eps * p).clamp(0.0, 1.0);
                });
            }
        }

        // -eps perturbation
        let mut alpha_minus: HashMap<String, Array2<f32>> = alphas.clone();
        for (name, pert) in &perturbations {
            if let Some(a) = alpha_minus.get_mut(name) {
                a.zip_mut_with(pert, |a_val, &p| {
                    *a_val = (*a_val - eps * p).clamp(0.0, 1.0);
                });
            }
        }

        // Evaluate with +eps
        let (bounds_plus, _) = graph_spec_crown_with_mul_binary_and_truncation(
            graph,
            input,
            spec_matrix,
            engine,
            None,
            None,
            None,
            Some(&alpha_plus),
            deadline,
            crown_backward_layers,
        )?;
        if deadline.is_some_and(|d| Instant::now() >= d) {
            debug!(
                "MulBinary SPSA: deadline exceeded after +eps evaluation at iteration {}/{}, returning current alphas",
                iter + 1,
                iterations
            );
            return Ok(optimized_any);
        }
        let lower_plus: f32 = bounds_plus.lower().iter().filter(|v| v.is_finite()).sum();

        // Evaluate with -eps
        let (bounds_minus, _) = graph_spec_crown_with_mul_binary_and_truncation(
            graph,
            input,
            spec_matrix,
            engine,
            None,
            None,
            None,
            Some(&alpha_minus),
            deadline,
            crown_backward_layers,
        )?;
        let lower_minus: f32 = bounds_minus.lower().iter().filter(|v| v.is_finite()).sum();

        let diff = lower_plus - lower_minus;

        // NaN guard: skip iteration if CROWN produced non-finite bounds.
        if !diff.is_finite() {
            debug!("MulBinary SPSA iter {}: non-finite diff, skipping", iter);
            continue;
        }

        // Adam update: gradient ascent (maximize lower bound → negate gradient).
        let t = (iter + 1) as f32;
        let bc1 = (1.0 - beta1.powf(t)).max(f32::EPSILON);
        let bc2 = (1.0 - beta2.powf(t)).max(f32::EPSILON);
        for (name, pert) in &perturbations {
            if let (Some(alpha), Some(m), Some(v)) = (
                alphas.get_mut(name),
                adam_m.get_mut(name),
                adam_v.get_mut(name),
            ) {
                alpha.zip_mut_with(pert, |a_val, &p| {
                    // This closure can't access m/v due to borrow rules, so we
                    // compute the gradient inline. For SPSA, the gradient for
                    // each parameter is: g_i = diff / (2 * eps * perturbation_i).
                    // We negate for gradient ascent.
                    let _ = p; // suppress unused warning (used below in indexed loop)
                    let _ = a_val;
                });

                // Use indexed loop instead of zip_mut_with to access m and v simultaneously.
                let shape = alpha.raw_dim();
                for row in 0..shape[0] {
                    for col in 0..shape[1] {
                        let p = pert[[row, col]];
                        let grad = diff / (2.0 * eps * p);
                        // Negate for gradient ascent (we want to maximize lower bound).
                        let neg_grad = -grad;
                        m[[row, col]] = beta1 * m[[row, col]] + (1.0 - beta1) * neg_grad;
                        v[[row, col]] = beta2 * v[[row, col]] + (1.0 - beta2) * neg_grad * neg_grad;
                        let m_hat = m[[row, col]] / bc1;
                        let v_hat = v[[row, col]] / bc2;
                        alpha[[row, col]] -= lr * m_hat / (v_hat.sqrt() + adam_eps);
                        alpha[[row, col]] = alpha[[row, col]].clamp(0.0, 1.0);
                        // NaN reset: if alpha becomes NaN, reset to 0.5.
                        if alpha[[row, col]].is_nan() {
                            alpha[[row, col]] = 0.5;
                            m[[row, col]] = 0.0;
                            v[[row, col]] = 0.0;
                        }
                    }
                }
                optimized_any = true;
            }
        }
    }

    Ok(optimized_any)
}
