// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PGD initialization helpers for graph verification.
//!
//! Extracted from `graph_pgd.rs` to stay under file size limit.

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{
    project_to_bounds_in_place, GraphNetwork, PgdConfig, PgdInitialization, PgdStepState,
};
use ny_tensor::BoundedTensor;

/// Evaluate a graph network at a concrete point (TRUE point forward).
///
/// Uses [`GraphNetwork::propagate_concrete_point`], NOT the whole-box
/// `propagate_ibp_with_engine`: for a point input the latter returns a
/// non-degenerate box (per-node soundness widening — esp. BatchNorm — amplified by
/// the deep DAG), so its `.lower()` deviates far from the network output and
/// fabricates false counterexamples that ONNX Runtime then rejects (cgan_2023
/// unknown-downgrade). The point forward matches ORT to ~1e-6. #cgan-eval.
pub(super) fn evaluate_graph(
    graph: &GraphNetwork,
    input: &ArrayD<f32>,
    gemm_engine: Option<&dyn GemmEngine>,
) -> Result<ArrayD<f32>> {
    // ORT-routed candidate scoring (#four-walls): milliseconds per forward vs
    // the internal per-layer walk on conv nets. Sound: attack outputs are only
    // ever violation CLAIMS; the independent vnncomp ORT gate re-confirms any
    // witness before a `sat` is scored. NY_ORT_ATTACK=0 disables.
    if let Some(output) = super::ort_attack::ort_forward_point(input) {
        return Ok(output);
    }
    let input_bounds = BoundedTensor::concrete(input.clone())?;
    let output_bounds = graph.propagate_concrete_point(&input_bounds, gemm_engine, None)?;
    Ok(output_bounds.center())
}

/// Independent concrete forward pass for counterexample re-validation.
///
/// Bypasses the engine used during PGD optimization by always using CPU-only
/// IBP (engine=None). This ensures the confirmation step is independent from
/// the evaluator used during the attack loop. Matches the
/// `independent_concrete_forward` pattern in `beta_crown/engine/pgd.rs:154`.
/// Part of #4419.
pub(super) fn independent_graph_forward(
    graph: &GraphNetwork,
    candidate: &ArrayD<f32>,
) -> Result<ArrayD<f32>> {
    let input_bounds = BoundedTensor::concrete(candidate.clone())?;
    // TRUE point forward (center-collapse per node), engine-free for independence.
    // See `evaluate_graph` for why the whole-box `.lower()` is wrong here. #cgan-eval.
    let output_bounds = graph.propagate_concrete_point(&input_bounds, None, None)?;
    Ok(output_bounds.center())
}

/// Re-validate a candidate counterexample via independent forward pass.
///
/// Returns `Some((candidate, revalidated_output))` if the counterexample is
/// confirmed, `None` (with a tracing warning) if re-validation fails.
/// Consolidates the repeated re-validation pattern in graph_pgd.rs and
/// graph_pgd_batched.rs. Part of #4419.
pub(super) fn revalidate_graph_counterexample(
    graph: &GraphNetwork,
    candidate: ArrayD<f32>,
    constraints: &[OutputConstraint],
    context: &str,
) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
    // With the ORT attack oracle active, re-validate against the same trusted
    // runtime semantics the final vnncomp gate uses — an internal-forward
    // re-check here would reject genuine ORT-confirmed violations whenever
    // ny's own conversion deviates (the cgan-class failure the trusted gate
    // exists for). The vnncomp gate still independently re-confirms every
    // emitted witness with a fresh session, so this cannot create a false
    // `sat` (#four-walls). Without the oracle, behavior is unchanged.
    let revalidated = match super::ort_attack::ort_forward_point(&candidate) {
        Some(output) => output,
        None => independent_graph_forward(graph, &candidate)?,
    };
    if super::check_unsafe_counterexample(&revalidated, constraints) {
        Ok(Some((candidate, revalidated)))
    } else {
        tracing::warn!("Graph PGD: {context} failed independent re-validation");
        Ok(None)
    }
}

/// Simple seeded PRNG (xorshift64) for PGD sampling without requiring rand crate.
pub(super) struct SimpleRng(u64);

impl SimpleRng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 & 0xFFFF_FFFF) as u32
    }

    pub(super) fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    pub(super) fn next_bool(&mut self) -> bool {
        self.next_u32() & 1 == 0
    }
}

/// Sample a uniform random point within input bounds.
pub(super) fn sample_uniform_point(input: &BoundedTensor, rng: &mut SimpleRng) -> ArrayD<f32> {
    let lower = input.lower();
    let upper = input.upper();
    let mut point = ArrayD::zeros(IxDyn(lower.shape()));
    for (p, (lo, hi)) in point.iter_mut().zip(lower.iter().zip(upper.iter())) {
        *p = lo + rng.next_f32() * (hi - lo);
    }
    point
}

/// The center of the input box — the nominal ("clean") input of an L∞
/// robustness spec, since the box is `[x0 - eps, x0 + eps]` about the clean
/// image `x0`. α,β-CROWN's PGD uses clean-init as its first restart: for a
/// robustly-trained net the counterexample is a *small* push from the
/// correctly-classified clean image, which uniform far-corner inits (measure
/// ~0 near the center in high dim) essentially never reach. Attack-only, so
/// this can only *find* counterexamples — it can never change a sound verdict.
pub(super) fn sample_center_point(input: &BoundedTensor) -> ArrayD<f32> {
    let lower = input.lower();
    let upper = input.upper();
    let mut point = ArrayD::zeros(IxDyn(lower.shape()));
    for (p, (lo, hi)) in point.iter_mut().zip(lower.iter().zip(upper.iter())) {
        *p = 0.5 * (lo + hi);
    }
    point
}

/// Initialize a graph PGD restart point according to the configured strategy.
///
/// - `Uniform`: standard `sample_uniform_point` (existing behavior).
/// - `Osi`: run `osi_steps` of SPSA-based signed gradient ascent on a random
///   output-space scalarization `<w, model(x)>` before returning the seed.
///
/// Reference: `OSI_init_C` in `attack_utils.py:328-362`.
pub(super) fn initialize_graph_point(
    pgd_config: &PgdConfig,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    rng: &mut SimpleRng,
    gemm_engine: Option<&dyn GemmEngine>,
) -> Result<ArrayD<f32>> {
    let mut x = sample_uniform_point(input, rng);
    if !matches!(pgd_config.initialization, PgdInitialization::Osi) || pgd_config.osi_steps == 0 {
        return Ok(x);
    }

    // Probe forward pass to discover output dimension.
    let probe_output = evaluate_graph(graph, &x, gemm_engine)?;
    let output_dim = probe_output.len();

    // Random output-space direction w in [-1, 1]^output_dim.
    let w: Vec<f32> = (0..output_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    let spsa_delta = pgd_config
        .suggested_spsa_delta(input)
        .max(pgd_config.spsa_delta);
    let mut step_state =
        PgdStepState::new_signed_gradient(pgd_config.alpha_mode, pgd_config.step_size, input);

    for _step in 0..pgd_config.osi_steps {
        // Rademacher perturbation vector.
        let delta: ArrayD<f32> =
            ArrayD::from_shape_fn(
                IxDyn(x.shape()),
                |_| {
                    if rng.next_bool() {
                        1.0
                    } else {
                        -1.0
                    }
                },
            );

        // Perturbed points, projected to bounds.
        let mut x_plus = &x + &(&delta * spsa_delta);
        let mut x_minus = &x - &(&delta * spsa_delta);
        project_to_bounds_in_place(&mut x_plus, input.lower(), input.upper());
        project_to_bounds_in_place(&mut x_minus, input.lower(), input.upper());

        let out_plus = evaluate_graph(graph, &x_plus, gemm_engine)?;
        let out_minus = evaluate_graph(graph, &x_minus, gemm_engine)?;

        let f_plus: f32 = out_plus.iter().zip(w.iter()).map(|(&o, &wi)| o * wi).sum();
        let f_minus: f32 = out_minus.iter().zip(w.iter()).map(|(&o, &wi)| o * wi).sum();

        if !f_plus.is_finite() || !f_minus.is_finite() {
            continue;
        }

        let diff = f_plus - f_minus;
        let sign_diff = if diff > 0.0 {
            1.0_f32
        } else if diff < 0.0 {
            -1.0_f32
        } else {
            0.0_f32
        };
        let pseudo_gradient = &delta * sign_diff;
        x = step_state.step(&pseudo_gradient, &x, input, true);
    }

    Ok(x)
}
