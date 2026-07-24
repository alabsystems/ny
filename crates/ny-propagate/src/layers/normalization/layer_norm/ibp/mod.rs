// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Interval bound propagation for LayerNorm.
//!
//! Decomposes the IBP implementation into focused modules:
//! - [`common`]: Shared helpers, validation, fallback bounds
//! - [`slices`]: Batch-prefix iteration and last-axis slice extraction
//! - [`forward_mode`]: Jacobian-based forward-mode IBP
//! - [`mean_only`]: MeanOnly interval arithmetic
//! - [`standard`]: Standard interval arithmetic with variance bounds

mod common;
mod forward_mode;
mod mean_only;
mod slices;
mod standard;

use std::borrow::Cow;

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_up_f32, BoundedTensor, L2Constraint};

use super::types::{LayerNormLayer, LayerNormMode};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

use common::validate_ibp_input;

impl LayerNormLayer {
    /// Build the per-slice L2 (Euclidean-ball) annotation for Standard LayerNorm.
    ///
    /// THE LEVER (LayerNorm case). The normalized part `z_i = (x_i − mean)/std`
    /// (std = sqrt(var + eps), var the population variance (1/n)·Σ(x_i − mean)²)
    /// satisfies the EXACT joint bound, in real arithmetic:
    ///   Σ_i z_i² = Σ(x_i − mean)² / (var + eps) = n·var / (var + eps) ≤ n,
    /// so ‖z‖₂ ≤ √n. The affine output is y = ny ⊙ z + beta, hence the ball is
    /// centred at `beta` (broadcast over slices) and
    ///   ‖y − beta‖₂ = ‖ny ⊙ z‖₂ ≤ (max_i|ny_i|)·√n.
    /// (The per-coordinate clamp uses √(n−1) from the zero-mean DOF; the L2 joint
    /// bound √n is a sound — slightly looser — superset, so it is safe to use.)
    ///
    /// FLOAT MARGIN identical in spirit to RMSNorm: a generous RELATIVE margin
    /// `(n + 4)·EPSILON`, every step rounded OUTWARD. Returns `None` (drop the
    /// annotation — sound) for MeanOnly mode (no variance normalization, so the
    /// √n bound does not hold), rank-0 / empty / huge axes, or non-finite params.
    fn compute_l2_constraint(&self, shape: &[usize]) -> Option<L2Constraint> {
        // THE GATE: only attach the sphere in a top-level plain IBP pass. Inside
        // iterative CROWN bound recomputation the gate is OFF, so we skip the
        // (per-pass) beta-centred center allocation entirely — byte-identical to
        // pre-lever and sound (the box bound is unchanged). See
        // `crate::l2_lever_gate`.
        if !crate::l2_lever_gate::l2_lever_active() {
            return None;
        }
        if self.mode != LayerNormMode::Standard {
            return None;
        }
        let ndim = shape.len();
        if ndim == 0 {
            return None;
        }
        let axis = ndim - 1;
        let norm_size = shape[axis];
        if norm_size == 0 || shape.contains(&0) {
            return None;
        }
        if self.ny.len() != norm_size || self.beta.len() != norm_size {
            return None;
        }
        let mut max_abs_g = 0.0_f32;
        for &g in self.ny.iter() {
            if !g.is_finite() {
                return None;
            }
            max_abs_g = max_abs_g.max(g.abs());
        }
        if self.beta.iter().any(|b| !b.is_finite()) {
            return None;
        }

        let nf = norm_size as f32;
        let sqrt_n = next_up_f32(nf.sqrt());
        let rel = next_up_f32((nf + 4.0) * f32::EPSILON);
        let one_plus_rel = next_up_f32(1.0 + rel);
        let radius = next_up_f32(next_up_f32(max_abs_g * sqrt_n) * one_plus_rel);
        if !radius.is_finite() {
            return None;
        }

        // Center = beta, broadcast across every normalization slice.
        let mut center = ArrayD::<f32>::zeros(IxDyn(shape));
        for mut lane in center.lanes_mut(ndarray::Axis(axis)) {
            for (slot, &b) in lane.iter_mut().zip(self.beta.iter()) {
                *slot = b;
            }
        }
        let radius_shape: Vec<usize> = shape[..axis].to_vec();
        let radius_arr = ArrayD::<f32>::from_elem(IxDyn(&radius_shape), radius);

        L2Constraint::new(center, radius_arr, axis, shape)
    }
}

impl BoundPropagation for LayerNormLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let ctx = validate_ibp_input(self, input)?;

        if self.mode == LayerNormMode::MeanOnly {
            return mean_only::propagate_interval(self, input, &ctx);
        }

        // Forward mode: use center point (midpoint) for mean/std computation.
        // This dramatically reduces bound explosion but is approximate for large perturbations.
        let out = if self.forward_mode {
            self.propagate_ibp_forward_mode(input)?
        } else {
            standard::propagate_interval(self, input, &ctx)?
        };

        // THE LEVER: attach the proven ‖y − beta‖₂ ≤ max|ny|·√n sphere so the
        // downstream Linear can swap its decorrelated box bound for the exact
        // Cauchy–Schwarz one. Intersection only tightens; drop on failure (sound).
        let shape = input.shape();
        Ok(match self.compute_l2_constraint(shape) {
            Some(c) => out.with_l2_constraint(c),
            None => out,
        })
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "LayerNorm is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        LayerNormLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}
