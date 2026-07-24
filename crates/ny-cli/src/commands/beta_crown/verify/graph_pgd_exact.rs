// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact graph PGD gradients via concrete CROWN linear extraction (#4274).
//!
//! Provides an exact-gradient fast path for the sequential graph PGD attack
//! on a narrow ResNet-style DAG whitelist. Falls back to SPSA when the
//! CROWN relaxation is not locally exact (tightness guard failure).
//! Reference: alpha-beta-CROWN `general_spec_attack.py:312-337`.

use anyhow::Result;
use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_propagate::{GraphNetwork, Layer};
use ny_tensor::BoundedTensor;
use std::time::Instant;

use super::graph_pgd::GraphPgdTarget;

// ---------------------------------------------------------------------------
// Exact-gradient whitelist
// ---------------------------------------------------------------------------

/// Narrow ResNet-style whitelist for layers whose concrete CROWN backward
/// pass produces an exact local Jacobian row (no relaxation gap at concrete
/// activation points). Distinct from the broader restart-batching whitelist.
fn layer_supports_exact_gradient(layer: &Layer) -> bool {
    matches!(
        layer,
        Layer::Conv2d(_)
            | Layer::Linear(_)
            | Layer::ReLU(_)
            | Layer::Add(_)
            | Layer::AveragePool(_)
            | Layer::Flatten(_)
            | Layer::Reshape(_)
            // GAN/deconv fragment (cgan etc.): ConvTranspose + BatchNorm are
            // affine/linear, exact at a point. Attack-only, so even an imperfect
            // point-VJP here cannot cause a false verdict (every CE is ORT-gated).
            | Layer::ConvTranspose1d(_)
            | Layer::ConvTranspose2d(_)
            | Layer::BatchNorm(_)
            // Affine constant arithmetic (cora_2024 MLP fragment: unfused Gemm =
            // MatMul + AddConstant bias, mnist Div-by-constant normalization;
            // d/dx (x/c) = 1/c). Exact at a point. Keep in lockstep with
            // `point_vjp_supported_fragment` in ny-propagate point_vjp.rs.
            | Layer::AddConstant(_)
            | Layer::SubConstant(_)
            | Layer::MulConstant(_)
            | Layer::DivConstant(_)
    )
}

pub(super) fn graph_supports_exact_gradients(graph: &GraphNetwork) -> bool {
    graph.node_names().iter().all(|node_name| {
        graph
            .node(node_name)
            .is_some_and(|node| layer_supports_exact_gradient(node.layer()))
    })
}

// ---------------------------------------------------------------------------
// Tightness tolerances
// ---------------------------------------------------------------------------

/// Maximum allowed gap between lower and upper CROWN coefficient rows.
const EXACT_GRAD_COEFF_TOL: f32 = 1e-4;
/// Maximum allowed gap between lower and upper concrete spec bounds.
const EXACT_GRAD_BOUND_TOL: f32 = 1e-4;
/// Maximum allowed deviation of the CROWN bound from the true concrete margin.
const EXACT_GRAD_MARGIN_TOL: f32 = 1e-3;

// ---------------------------------------------------------------------------
// Core: exact gradient extraction
// ---------------------------------------------------------------------------

/// Attempt to compute an exact gradient for the given target at concrete point `x`
/// using spec-guided CROWN with linear extraction. Returns `Ok(Some(grad))` when
/// the extracted linear row is locally exact (within tolerances), `Ok(None)` when
/// the graph doesn't support exact gradients or the relaxation is too loose, and
/// `Err(...)` only on internal failures.
///
/// Algorithm (design: `designs/2026-03-20-issue-4274-graph-pgd-exact-gradients.md`):
/// 1. Build concrete input bounds from `x`.
/// 2. Collect per-node IBP bounds at the concrete point.
/// 3. Build a single-row spec matrix from the target.
/// 4. Run spec-guided CROWN with node bounds + linear extraction.
/// 5. Validate tightness: coeff gap, bound gap, and margin error.
/// 6. Return the gradient row reshaped to `x.shape()`.
pub(super) fn exact_graph_margin_gradient(
    graph: &GraphNetwork,
    x: &ArrayD<f32>,
    output: &ArrayD<f32>,
    target: &GraphPgdTarget,
    num_outputs: usize,
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<Option<ArrayD<f32>>> {
    let spec_row = target.to_spec_row(num_outputs);
    exact_graph_spec_gradient(graph, x, output, &spec_row, gemm_engine, deadline)
}

/// Exact gradient of an arbitrary single-row linear output functional
/// `spec_row · y(x)` at the concrete point `x`.
///
/// The JOINT AND-clause attack objective (#soundnessbench) is such a row: the
/// sum of the hinge-active conjuncts' margin rows. Constant hinge offsets drop
/// out of the derivative, so ONE backward pass yields the exact joint gradient
/// — no per-conjunct passes needed.
///
/// The tightness guards on the certified fallback compare the CROWN spec bound
/// against the concretely evaluated LINEAR value `spec_row · output` (not
/// `target.margin`, which for Constant/NegConstant targets is shifted by the
/// threshold and made the guard spuriously reject exact rows).
pub(super) fn exact_graph_spec_gradient(
    graph: &GraphNetwork,
    x: &ArrayD<f32>,
    output: &ArrayD<f32>,
    spec_row: &Array2<f32>,
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<Option<ArrayD<f32>>> {
    if !graph_supports_exact_gradients(graph) {
        return Ok(None);
    }

    // Fast attack gradient (DEFAULT ON; opt out with `NY_PGD_FAST_GRAD=0`): a plain
    // point-Jacobian VJP that skips the certified two-sided relaxation / coeff-error
    // carrier / tightness guards. Validated to match this certified path to ~1e-3
    // (ny-propagate point_vjp gradient-check) AND finite differences to ~1e-2.
    // Strictly better than the certified path for an attack: ~same-or-faster AND it
    // never spuriously falls back to noisy SPSA when the deep-resnet tightness
    // guards fail. Attack-only ⇒ soundness-irrelevant (every SAT is ORT-re-validated
    // by revalidate_graph_counterexample + the vnncomp gate). Measured: lifts 13/14
    // cifar100_resnet_medium sat instances that were 0/14 before. Falls through to
    // the certified path on None (non-whitelist fragment / deadline).
    if std::env::var("NY_PGD_FAST_GRAD").ok().as_deref() != Some("0") {
        if let Some(g) = graph.attack_point_gradient(x, spec_row, gemm_engine, deadline)? {
            return Ok(Some(g));
        }
    }

    let input_bounds = BoundedTensor::concrete(x.clone())?;

    let diag2 = std::env::var("NY_PGD_DIAG2").ok().as_deref() == Some("1");
    let t_nb0 = Instant::now();
    let node_bounds =
        graph.collect_node_bounds_with_engine_and_deadline(&input_bounds, gemm_engine, deadline)?;
    let t_nb = t_nb0.elapsed();

    let t_cr0 = Instant::now();
    let (spec_bounds, linear) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            &input_bounds,
            spec_row,
            gemm_engine,
            &node_bounds,
            deadline,
        )?;
    if diag2 {
        eprintln!(
            "[grad-timing] collect_node_bounds={:.1}ms  crown_propagate={:.1}ms",
            t_nb.as_secs_f64() * 1e3,
            t_cr0.elapsed().as_secs_f64() * 1e3
        );
    }

    let linear = match linear {
        Some(lb) => lb,
        None => return Ok(None),
    };

    // Validate: spec bounds must be finite
    let spec_lo = spec_bounds.lower()[[0]];
    let spec_hi = spec_bounds.upper()[[0]];
    if !spec_lo.is_finite() || !spec_hi.is_finite() {
        return Ok(None);
    }

    // Validate: coefficient matrices must be finite
    if !linear.lower_a().iter().all(|v| v.is_finite())
        || !linear.upper_a().iter().all(|v| v.is_finite())
    {
        return Ok(None);
    }

    // Tightness guard: gap between lower and upper coefficient rows
    let coeff_gap: f32 = (linear.upper_a() - linear.lower_a()).mapv(f32::abs).sum();
    if coeff_gap > EXACT_GRAD_COEFF_TOL {
        return Ok(None);
    }

    // Tightness guard: gap between lower and upper concrete bounds
    let bound_gap = (spec_hi - spec_lo).abs();
    if bound_gap > EXACT_GRAD_BOUND_TOL {
        return Ok(None);
    }

    // Tightness guard: bound error vs the concretely evaluated linear spec
    // value at the point (constant margin offsets excluded on BOTH sides, so
    // the comparison is exact for Constant/NegConstant targets too).
    let true_value: f32 = output
        .iter()
        .zip(spec_row.row(0).iter())
        .map(|(&y, &c)| y * c)
        .sum();
    let margin_err = (spec_lo - true_value)
        .abs()
        .max((spec_hi - true_value).abs());
    if margin_err > EXACT_GRAD_MARGIN_TOL {
        return Ok(None);
    }

    // Extract gradient: use the average of lower_a and upper_a (they should be
    // nearly identical at a concrete point where all ReLUs are decided).
    let grad_row = (linear.lower_a() + linear.upper_a()) * 0.5;
    let grad_flat: Vec<f32> = grad_row.row(0).to_vec();
    let grad = ArrayD::from_shape_vec(IxDyn(x.shape()), grad_flat)?;

    Ok(Some(grad))
}
