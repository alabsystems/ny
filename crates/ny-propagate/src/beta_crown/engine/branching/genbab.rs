// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GenBaB nonlinear split selection helpers.

use super::*;

impl BetaCrownVerifier {
    pub(in crate::beta_crown::engine) fn select_genbab_branch(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        split_nodes: &[String],
        genbab: &NonlinearBranching,
    ) -> Result<Option<BranchingDecision>> {
        let mut node_bounds: std::collections::HashMap<String, BoundedTensor> = domain
            .node_bounds
            .iter()
            .map(|(k, v)| (k.clone(), v.as_ref().clone()))
            .collect();
        // Include the network input so RmsNorm branching can read input bounds
        // when the norm's input is the network input (#norm-genbab).
        node_bounds.insert(
            NETWORK_INPUT.to_string(),
            domain.input_bounds.as_ref().clone(),
        );

        // Thread the domain's already-applied norm inv_rms windows so RmsNorm
        // branching computes the effective (post-constraint) per-group range and
        // converges (#norm-genbab).
        let norm_windows = domain.history().norm_inv_rms_overrides();
        let decisions = genbab.decisions_with_norm_windows(
            graph,
            &node_bounds,
            split_nodes,
            norm_windows.as_ref(),
        )?;
        Ok(decisions.into_iter().next())
    }

    /// Find splittable nonlinear neurons (GenBaB).
    ///
    /// For elementwise activations: checks first input's bounds for width.
    /// For BilinearCrown: checks BOTH inputs' bounds (Q and K), since splitting
    /// either input reduces McCormick overapproximation.
    ///
    /// See designs/2026-01-28-genbab-branching.md and
    /// designs/2026-03-04-286-attention-bilinear-alternative.md Approach C.
    pub(in crate::beta_crown::engine) fn find_splittable_graph_nodes(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        nonlinear_nodes: &[String],
        genbab: &NonlinearBranching,
    ) -> Vec<String> {
        let mut splittable = Vec::new();
        let min_width = genbab.config().min_branch_width;
        // Already-applied norm inv_rms windows on this domain's BaB path
        // (#norm-genbab); used to compute effective per-group RmsNorm widths.
        let norm_windows = domain.history().norm_inv_rms_overrides();

        for node_name in nonlinear_nodes {
            let node = match graph.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };

            if !genbab.is_splittable(&node.layer) {
                continue;
            }

            // RmsNorm (#norm-genbab): splittable when the WIDEST group's EFFECTIVE
            // inv_rms range (input-derived, intersected with any window already
            // applied to that group on the BaB path) exceeds min_branch_width.
            // Intersecting with the applied windows is essential for convergence:
            // norm splits don't narrow input bounds, so without it the node would
            // report splittable forever.
            if let Layer::RmsNorm(rms) = &node.layer {
                let input_name = match node.inputs.first() {
                    Some(s) => s.as_str(),
                    None => continue,
                };
                let bounds: &BoundedTensor = if input_name == NETWORK_INPUT {
                    domain.input_bounds.as_ref()
                } else {
                    match domain.node_bounds.get(input_name) {
                        Some(b) => b.as_ref(),
                        None => continue,
                    }
                };
                let applied = norm_windows.as_ref().and_then(|m| m.get(node_name));
                if rms_norm_max_group_width(bounds, rms.eps, applied.map(|v| v.as_slice()))
                    >= min_width
                {
                    splittable.push(node_name.clone());
                }
                continue;
            }

            // BilinearCrown and MulBinary: both are McCormick-relaxed bilinear ops
            // with two perturbed inputs. Check BOTH inputs for splittable neurons,
            // since splitting either input (x/y, or Q/K) reduces the McCormick
            // envelope gap (ux−lx)(uy−ly)/4.
            // Reference: auto_LiRPA get_split_nodes (beta_crown.py:62-79) adds both
            // input nodes as split candidates.
            if matches!(node.layer, Layer::BilinearCrown(_) | Layer::MulBinary(_)) {
                let mut found = false;
                for input_name in &node.inputs {
                    let bounds: &BoundedTensor = if input_name == NETWORK_INPUT {
                        domain.input_bounds.as_ref()
                    } else {
                        match domain.node_bounds.get(input_name.as_str()) {
                            Some(b) => b.as_ref(),
                            None => continue,
                        }
                    };
                    let flat = bounds.flatten();
                    for neuron_idx in 0..flat.len() {
                        let l = flat.lower()[[neuron_idx]];
                        let u = flat.upper()[[neuron_idx]];
                        if (u - l) >= min_width && l.is_finite() && u.is_finite() {
                            found = true;
                            break;
                        }
                    }
                    if found {
                        break;
                    }
                }
                if found {
                    splittable.push(node_name.clone());
                }
                continue;
            }

            // Elementwise activations: check first input's bounds.
            // #2098: Skip nodes with empty inputs instead of fabricating NETWORK_INPUT.
            let pre_name = match node.inputs.first() {
                Some(s) => s.as_str(),
                None => {
                    tracing::warn!(node = %node_name, "activation node has empty inputs — skipping");
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
                let l = flat.lower()[[neuron_idx]];
                let u = flat.upper()[[neuron_idx]];
                let width = u - l;

                if width >= min_width {
                    // ReLU and Sign are zero-threshold activations: only
                    // unstable neurons (crossing 0) are splittable.
                    // Other activations are splittable on width alone.
                    if matches!(node.layer, Layer::ReLU(_) | Layer::Sign(_)) {
                        if l < 0.0 && u > 0.0 {
                            splittable.push(node_name.clone());
                            break;
                        }
                    } else {
                        splittable.push(node_name.clone());
                        break;
                    }
                }
            }
        }

        splittable
    }
}

/// Maximum over normalization groups of the EFFECTIVE `inv_rms` interval width
/// for a `Layer::RmsNorm` node (#norm-genbab).
///
/// Each group's IBP-derived `inv_rms` interval is intersected with the window
/// already applied to that group (`applied_windows[g]`, if any) before its width
/// is taken. Returning the per-group MAX (not the union) matches the selector,
/// which splits the single widest group; intersecting with applied windows makes
/// the width shrink each split so the node stops being splittable once every
/// group is narrow. Returns 0.0 on degenerate/empty input.
fn rms_norm_max_group_width(
    bounds: &BoundedTensor,
    eps: f32,
    applied_windows: Option<&[Option<(f32, f32)>]>,
) -> f32 {
    use crate::layers::normalization::math_common::square_interval_bounds;
    use ny_tensor::{next_down_f32, next_up_f32};
    let shape = bounds.shape();
    let norm_size = match shape.last().copied() {
        Some(n) if n > 0 => n,
        _ => return 0.0,
    };
    let flat_l = crate::contiguous_flat_slice(bounds.lower());
    let flat_u = crate::contiguous_flat_slice(bounds.upper());
    if !flat_l.len().is_multiple_of(norm_size) {
        return 0.0;
    }
    let groups = flat_l.len() / norm_size;
    let nf = norm_size as f64;
    let mut max_width = 0.0f32;
    for g in 0..groups {
        let xl = &flat_l[g * norm_size..(g + 1) * norm_size];
        let xu = &flat_u[g * norm_size..(g + 1) * norm_size];
        let mut var_l = 0.0f64;
        let mut var_u = 0.0f64;
        for i in 0..norm_size {
            let (sq_l, sq_u) = square_interval_bounds(xl[i], xu[i]);
            var_l += sq_l as f64;
            var_u += sq_u as f64;
        }
        let var_l = next_down_f32((var_l / nf) as f32);
        let var_u = next_up_f32((var_u / nf) as f32);
        let var_eps_l = next_down_f32((var_l as f64 + eps as f64) as f32);
        let var_eps_u = next_up_f32((var_u as f64 + eps as f64) as f32);
        let rms_l = next_down_f32((var_eps_l as f64).sqrt() as f32);
        let rms_u = next_up_f32((var_eps_u as f64).sqrt() as f32);
        let mut lo = next_down_f32(1.0 / rms_u);
        let mut hi = next_up_f32(1.0 / rms_l);
        if let Some((wlo, whi)) = applied_windows.and_then(|w| w.get(g).copied().flatten()) {
            lo = lo.max(wlo);
            hi = hi.min(whi);
        }
        if !lo.is_finite() || !hi.is_finite() || hi < lo {
            continue;
        }
        max_width = max_width.max(hi - lo);
    }
    max_width
}
