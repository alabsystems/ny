// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #binding-row-replay (DARK — no production callers; consult #6 arbitration,
//! `docs/ADJOINT_PORT_CRITIQUE_CONSULT6_2026-08-01.md` §4).
//!
//! CPU prototype of the TRUE per-neuron alpha gradient for ONE spec row
//! ("binding row") of the certified DAG alpha fold:
//!
//! ```text
//!   d lb_row / d alpha_k[i] = nu_k[i] * hhat_k[i](x*)     when nu_k[i] > 0
//!                           = 0                            otherwise
//! ```
//!
//! where `nu_k = intermediate.a_at_relu[k].row(row)` (the row's coefficients
//! on ReLU k's OUTPUT when the backward reached it), `x*` is the row's
//! box-corner concretization argmin read off `intermediate.final_bounds`
//! (`concretize_scalar_f64`: `la > 0 -> xl`, `la < 0 -> xu`), and
//! `hhat_k(x*)` is ONE relaxed point forward through the graph applying, at
//! every ReLU, exactly the per-neuron affine substitution the backward
//! selected for this row (`propagate_linear_with_alpha_impl`,
//! `layers/activations/relu/mod.rs`):
//!
//! - `nu > 0`: slope `alpha_row[i]` (1 if `l >= 0`, 0 if `u <= 0`), intercept 0;
//! - `nu < 0`: the upper chord `(lambda, lambda_intercept)` including the
//!   walk's non-finite-endpoint arms;
//! - `nu == 0`: slope 0, intercept 0 (the walk's `la == 0` arm leaves the
//!   composed coefficient at zero and folds no intercept).
//!
//! This replaces the LOCAL rule's `pre_lower[i]` substitution
//! (`backward/gradients.rs`), which the FD oracle
//! (`backward/true_grad_oracle_tests.rs`) proved sign-WRONG whenever the
//! relaxed pre-activation at `x*` is positive.
//!
//! Every non-ReLU node replays through its existing IBP dispatch on a
//! degenerate (point) box, collapsed back to the interval midpoint after each
//! node so per-node outward rounding cannot compound (the
//! `collect_node_activations_pointwise` idiom). Cost per binding row =
//! O(one concrete forward) + O(sum of ReLU widths); no dense per-neuron
//! input-affine map is ever reconstructed.
//!
//! Scope/limits of the prototype (typed errors, never a silent wrong answer):
//! - requires the DENSE `a_at_relu` capture (the AnalyticChain intermediates
//!   pass); refuses beta-sparse or missing A-matrices;
//! - alpha at replayed ReLUs is per-neuron (`full_conv_alpha` default) OR
//!   channel-shared (#channel-alpha-grad, `full_conv_alpha: false`): a
//!   channel-shared α (length C over a C·H·W node, keyed on the node's
//!   MEASURED recorded geometry, never a config flag) replays via the
//!   broadcast slope α_c at every spatial position, and the returned gradient
//!   is the exact chain rule `dL/dα_c = Σ_{h,w} ν_{c,h,w}·ĥ_{c,h,w}(x*)`
//!   (length C — the layout `update_all_alphas` consumes). Widths that are
//!   neither per-neuron nor channel-reconcilable are refused;
//! - `binding_row` indexes the SEED row space of the capturing fold (identical
//!   to the output row space unless a #margin-subset-alpha scope was
//!   published, in which case the caller owns the compact mapping);
//! - lower path only (the margin objective ascends spec-row LOWER bounds).

use std::collections::HashMap;

use ndarray::{Array1, ArrayD};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::{GraphAlphaCrownIntermediate, GraphAlphaState};
use crate::layers::activations::relu_crossing_upper_chord;
use crate::layers::{BoundPropagation, Layer};
use crate::network::core::{GraphNetwork, NETWORK_INPUT};

/// Output of one binding-row replay.
#[derive(Debug)]
#[allow(dead_code)] // Diagnostic receipt retains replay components for audit output.
pub(crate) struct BindingRowReplay {
    /// TRUE d(lb_row)/d(alpha) per ReLU node, at the node's ALPHA width —
    /// full neuron width for per-neuron α, channel width C for channel-shared
    /// α (spatial positions channel-summed, #channel-alpha-grad) — zeros at
    /// stable / non-positive-nu / non-finite entries (AnalyticChain mask
    /// conventions).
    pub(crate) grads: HashMap<String, Array1<f32>>,
    /// Relaxed pre-activation at `x*` per ReLU node (diagnostics + tests).
    pub(crate) hhat: HashMap<String, Array1<f64>>,
    /// The row's concretization argmin corner.
    pub(crate) x_star: ArrayD<f32>,
    /// The replayed network output value at `binding_row` — in exact
    /// arithmetic this equals the certified fold's lower bound for the row
    /// (modulo the fold's directed rounding), which tests exploit as a
    /// whole-replay parity check.
    pub(crate) replayed_row_value: f64,
}

/// The walk's per-neuron LOWER-row affine substitution `(slope, intercept)`,
/// mirroring `propagate_linear_with_alpha_impl` arm-for-arm (including its
/// non-finite endpoint handling) for the branch selected by `nu`'s sign.
pub(super) fn lower_row_slope_intercept(
    l: f32,
    u: f32,
    nu: f32,
    alpha: Option<f32>,
) -> Result<(f32, f32)> {
    if nu == 0.0 {
        // `la == 0` arm: composed coefficient stays zero, no intercept folded.
        return Ok((0.0, 0.0));
    }
    if nu > 0.0 {
        // `la > 0` arm: slope alpha_lower_i, intercept 0.
        let s = if l >= 0.0 {
            1.0
        } else if u <= 0.0 {
            0.0
        } else {
            alpha.ok_or_else(|| {
                NyError::InvalidSpec(
                    "#binding-row-replay: crossing neuron with nu > 0 needs an alpha value"
                        .to_string(),
                )
            })?
        };
        return Ok((s, 0.0));
    }
    // `la < 0` arm: the upper envelope (lambda, lambda_intercept), exactly
    // the walk's lambda-loop arms.
    Ok(if l.is_nan() || u.is_nan() {
        (0.0, f32::INFINITY)
    } else if l >= 0.0 {
        (1.0, 0.0)
    } else if u <= 0.0 {
        (0.0, 0.0)
    } else if l.is_infinite() && u.is_infinite() {
        (0.0, f32::INFINITY)
    } else if u.is_infinite() {
        (1.0, -l)
    } else if l.is_infinite() {
        (0.0, u)
    } else {
        relu_crossing_upper_chord(l, u, None)
    })
}

/// Degenerate (point) box around `v`.
pub(super) fn point_box(v: &ArrayD<f32>) -> Result<BoundedTensor> {
    BoundedTensor::new(v.clone(), v.clone())
}

/// Interval midpoint, collapsing per-node sound outward rounding back to a
/// point so it cannot compound down the replay.
pub(super) fn midpoint(bt: &BoundedTensor) -> ArrayD<f32> {
    let mut mid = bt.lower().clone();
    mid.zip_mut_with(bt.upper(), |l, &u| {
        let l64 = *l as f64;
        *l = (l64 + 0.5 * (u as f64 - l64)) as f32;
    });
    mid
}

impl GraphNetwork {
    /// TRUE d(binding-row lower bound)/d-alpha by stored-intermediate readout
    /// plus one relaxed point forward. See module docs for the contract.
    ///
    /// `intermediate` must come from the SAME fold state (`alpha_state`,
    /// node-bounds map) the caller is differentiating —
    /// `dag_alpha_backward_pass_with_intermediates` output at the current
    /// alpha iterate.
    pub(crate) fn binding_row_true_alpha_grads(
        &self,
        input: &BoundedTensor,
        alpha_state: &GraphAlphaState,
        intermediate: &GraphAlphaCrownIntermediate,
        binding_row: usize,
    ) -> Result<BindingRowReplay> {
        // === 1. x*: sign readout of the row's final input-affine coefficients ===
        let final_a = intermediate.final_bounds.lower_a();
        if binding_row >= final_a.nrows() {
            return Err(NyError::InvalidSpec(format!(
                "#binding-row-replay: binding_row {} out of range ({} seed rows in final_bounds)",
                binding_row,
                final_a.nrows()
            )));
        }
        let input_flat = input.flatten();
        let n_in = input_flat.len();
        if final_a.ncols() != n_in {
            return Err(NyError::ShapeMismatch {
                expected: vec![n_in],
                got: vec![final_a.ncols()],
            });
        }
        let xl = input_flat.lower();
        let xu = input_flat.upper();
        let mut xs: Vec<f32> = Vec::with_capacity(n_in);
        for j in 0..n_in {
            let a = final_a[[binding_row, j]];
            // concretize_scalar_f64 convention: la > 0 pays xl, la < 0 pays xu.
            // A zero coefficient contributes nothing either way; the midpoint
            // is the central-difference subgradient at the (measure-zero) kink.
            xs.push(if a > 0.0 {
                xl[[j]]
            } else if a < 0.0 {
                xu[[j]]
            } else {
                let l = xl[[j]] as f64;
                (l + 0.5 * (xu[[j]] as f64 - l)) as f32
            });
        }
        let x_star = ArrayD::from_shape_vec(input.lower().raw_dim(), xs).map_err(|e| {
            NyError::InvalidSpec(format!("#binding-row-replay: x* reshape failed: {e}"))
        })?;

        // === 2. Relaxed point forward at x* ===
        let exec_order = self.exec_order()?;
        let mut values: HashMap<&str, ArrayD<f32>> = HashMap::new();
        let mut hhat: HashMap<String, Array1<f64>> = HashMap::new();

        let value_of = |values: &HashMap<&str, ArrayD<f32>>,
                        name: &str,
                        x_star: &ArrayD<f32>|
         -> Result<ArrayD<f32>> {
            if name == NETWORK_INPUT {
                return Ok(x_star.clone());
            }
            values.get(name).cloned().ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "#binding-row-replay: value for input node '{name}' not computed yet"
                ))
            })
        };

        for node_name in exec_order {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {node_name}")))?;

            let out: ArrayD<f32> = if matches!(node.layer, Layer::ReLU(_)) {
                let src = node.inputs.first().map(String::as_str).ok_or_else(|| {
                    NyError::InvalidSpec(format!("ReLU node '{node_name}' has no input"))
                })?;
                let z = value_of(&values, src, &x_star)?;
                let n = z.len();

                let (pre_l, pre_u) = intermediate.pre_relu_bounds(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "#binding-row-replay: no pre-ReLU bounds captured for '{node_name}' \
                             — intermediate must come from the AnalyticChain intermediates fold"
                    ))
                })?;
                let a_mat = intermediate.a_at_relu(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "#binding-row-replay: no DENSE a_at_relu for '{node_name}' (beta-sparse \
                         capture or truncated fold) — the replay refuses partial state"
                    ))
                })?;
                if pre_l.len() != n || a_mat.ncols() != n || binding_row >= a_mat.nrows() {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![n, n, binding_row + 1],
                        got: vec![pre_l.len(), a_mat.ncols(), a_mat.nrows()],
                    });
                }
                // Effective per-row LOWER alpha (#spec-axis-alpha aware; falls
                // back to the shared vector bit-identically when no slot owns
                // the row). None => no alpha stored for this node: legal as
                // long as no crossing neuron needs it.
                let row_alpha = alpha_state.alpha_for_row(node_name, binding_row);
                let alpha_arr = row_alpha.as_ref().map(|r| r.as_array());
                // #channel-alpha-grad: α index stride. `1` reads α per-neuron;
                // `spatial` reads the channel-shared α_c broadcast at every
                // spatial position of channel c = i / spatial. Keyed on the
                // MEASURED shapes (stored α length, recorded [C,H,W] geometry,
                // neuron width) — never a config flag; anything else refuses.
                let alpha_stride: usize = match alpha_arr {
                    None => 1,
                    Some(a) if a.len() == n => 1,
                    Some(a) => {
                        let Some((_channels, spatial)) =
                            alpha_state.channel_reduction_geometry(node_name, a.len(), n)
                        else {
                            return Err(NyError::UnsupportedConfiguration(format!(
                                "#binding-row-replay: alpha width {} at '{node_name}' is neither \
                                 per-neuron (width {}) nor channel-shared over the node's \
                                 recorded conv geometry — refusing to guess a layout",
                                a.len(),
                                n
                            )));
                        };
                        spatial
                    }
                };

                let z_flat: Vec<f64> = z.iter().map(|&v| v as f64).collect();
                let mut h = ArrayD::<f32>::zeros(z.raw_dim());
                for (i, hv) in h.iter_mut().enumerate() {
                    let nu = a_mat[[binding_row, i]];
                    let (s, t) = lower_row_slope_intercept(
                        pre_l[i],
                        pre_u[i],
                        nu,
                        alpha_arr.map(|a| a[i / alpha_stride]),
                    )?;
                    *hv = (s as f64 * z_flat[i] + t as f64) as f32;
                }
                hhat.insert(node_name.clone(), Array1::from(z_flat));
                h
            } else {
                let bt = match node.inputs.len() {
                    1 => {
                        let a = value_of(&values, &node.inputs[0], &x_star)?;
                        node.layer.propagate_ibp(&point_box(&a)?)?
                    }
                    2 => {
                        let a = value_of(&values, &node.inputs[0], &x_star)?;
                        let b = value_of(&values, &node.inputs[1], &x_star)?;
                        node.layer
                            .propagate_ibp_binary(&point_box(&a)?, &point_box(&b)?)?
                    }
                    3 => {
                        let a = value_of(&values, &node.inputs[0], &x_star)?;
                        let b = value_of(&values, &node.inputs[1], &x_star)?;
                        let c = value_of(&values, &node.inputs[2], &x_star)?;
                        node.layer.propagate_ibp_ternary(
                            &point_box(&a)?,
                            &point_box(&b)?,
                            &point_box(&c)?,
                        )?
                    }
                    k => {
                        return Err(NyError::UnsupportedOp(format!(
                            "#binding-row-replay: {k}-ary node '{node_name}' \
                             ({}) not supported by the prototype",
                            node.layer.layer_type()
                        )));
                    }
                };
                midpoint(&bt)
            };
            values.insert(node_name.as_str(), out);
        }

        // === 3. Combine: grad_i = nu_i * hhat_i(x*) under the walk's masks ===
        //
        // #channel-alpha-grad: for a channel-shared α node the output is at
        // ALPHA width C, accumulating grad[c] += ν_i·ĥ_i(x*) over that
        // channel's spatial positions i (the same per-position masks as the
        // per-neuron path) — the exact chain rule through the α_c broadcast.
        let mut grads: HashMap<String, Array1<f32>> = HashMap::new();
        for (name, (pre_l, pre_u)) in &intermediate.pre_relu_bounds {
            let n = pre_l.len();
            // Alpha-width output + spatial stride, keyed on MEASURED shapes
            // exactly like the replay arm above (which already refused any
            // irreconcilable layout before reaching this combine).
            let (alpha_len, stride) = match alpha_state.alpha(name) {
                Some(a) if a.len() != n => {
                    match alpha_state.channel_reduction_geometry(name, a.len(), n) {
                        Some((channels, spatial)) => (channels, spatial),
                        None => (n, 1),
                    }
                }
                _ => (n, 1),
            };
            let mut g = Array1::<f32>::zeros(alpha_len);
            if let (Some(a_mat), Some(z)) = (intermediate.a_at_relu(name), hhat.get(name)) {
                for i in 0..n {
                    let l = pre_l[i];
                    let u = pre_u[i];
                    // AnalyticChain mask conventions (backward/gradients.rs):
                    // finite bounds, strictly unstable, finite positive nu.
                    if !l.is_finite() || !u.is_finite() {
                        continue;
                    }
                    if l >= 0.0 || u <= 0.0 {
                        continue;
                    }
                    let nu = a_mat[[binding_row, i]];
                    if !nu.is_finite() || nu <= 0.0 {
                        continue;
                    }
                    if stride == 1 {
                        // Per-neuron: plain assignment, bit-identical to the
                        // historical path (`+=` would rewrite -0.0 to +0.0).
                        g[i] = (nu as f64 * z[i]) as f32;
                    } else {
                        g[i / stride] += (nu as f64 * z[i]) as f32;
                    }
                }
            }
            grads.insert(name.clone(), g);
        }

        // Replayed output value for the row (parity diagnostics).
        let output_name = if self.output_node.is_empty() {
            exec_order.last().map(String::as_str).unwrap_or_default()
        } else {
            self.output_node.as_str()
        };
        let replayed_row_value = values
            .get(output_name)
            .and_then(|v| v.iter().nth(binding_row))
            .map(|&v| v as f64)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "#binding-row-replay: output node '{output_name}' missing row {binding_row}"
                ))
            })?;

        Ok(BindingRowReplay {
            grads,
            hhat,
            x_star,
            replayed_row_value,
        })
    }
}

#[cfg(test)]
mod tests;
