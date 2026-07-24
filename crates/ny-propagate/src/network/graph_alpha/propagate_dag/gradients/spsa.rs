// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SPSA gradient computation for DAG α-CROWN.
//!
//! Extracted from `gradients.rs` as part of #4187.

use super::super::super::runtime_state::DagAlphaRuntimeState;
use super::super::DagAlphaLoopContext;
use super::{alpha_refs, zeros_like_gradients};
use crate::network::alpha_crown_loop::finite_lower_sum;
use crate::network::core::GraphNetwork;
use ndarray::{Array1, Array2, Array4};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;

impl GraphNetwork {
    /// SPSA gradient computation for DAG α-CROWN.
    ///
    /// Uses simultaneous perturbation stochastic approximation to estimate gradients.
    /// Modifies bilinear/mul_binary alpha maps (snapshot, perturb, restore) and
    /// accumulates into the provided gradient maps.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::network::graph_alpha::propagate_dag) fn compute_spsa_gradients(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &mut DagAlphaRuntimeState,
        bilinear_alphas: &mut HashMap<String, Array4<f32>>,
        mul_binary_alphas: &mut HashMap<String, Array2<f32>>,
        bilinear_grads: &mut HashMap<String, Array4<f32>>,
        mul_binary_grads: &mut HashMap<String, Array2<f32>>,
        gradients: &[Array1<f32>],
        num_relus: usize,
        eps: f32,
    ) -> Result<Vec<Array1<f32>>> {
        use rand::RngExt;
        let mut rng = crate::random::rng();

        let mut avg_grads: Vec<Array1<f32>> = (0..num_relus)
            .map(|relu_idx| {
                runtime
                    .relu_len(relu_idx)
                    .map(Array1::zeros)
                    .ok_or_else(|| {
                        NyError::InternalError(format!(
                            "missing DAG ReLU alpha length for index {relu_idx}"
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        let original_graph = runtime.snapshot_graph();
        // Save bilinear originals for restoration after SPSA perturbation (#3287).
        let original_bilinear: HashMap<String, Array4<f32>> = bilinear_alphas.clone();
        // Save MulBinary originals (#3439 Phase 3).
        let original_mul_binary: HashMap<String, Array2<f32>> = mul_binary_alphas.clone();

        // Scratch gradient buffers (required by DAG pass signature).
        let mut scratch = zeros_like_gradients(gradients);
        let mut scratch_upper = zeros_like_gradients(gradients);

        // Pre-allocate perturbation buffers once; refill in place each sample (#2297).
        let mut perturbations: Vec<Array1<f32>> = (0..num_relus)
            .map(|relu_idx| {
                runtime
                    .relu_len(relu_idx)
                    .map(Array1::zeros)
                    .ok_or_else(|| {
                        NyError::InternalError(format!(
                            "missing DAG ReLU alpha length for index {relu_idx}"
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut bilinear_perts: HashMap<String, Array4<f32>> = bilinear_alphas
            .iter()
            .map(|(name, alpha)| (name.clone(), Array4::zeros(alpha.raw_dim())))
            .collect();
        let mut mul_binary_perts: HashMap<String, Array2<f32>> = mul_binary_alphas
            .iter()
            .map(|(name, alpha)| (name.clone(), Array2::zeros(alpha.raw_dim())))
            .collect();

        // Wrap loop body in closure so `?` returns to the closure,
        // not the outer function. Ensures alpha restoration runs
        // unconditionally, even on CROWN propagation error (#2554).
        let result = (|| -> Result<()> {
            for _sample in 0..ctx.config.spsa_samples {
                // Refill ReLU perturbations in place (#2297).
                for relu_idx in 0..num_relus {
                    let mask = runtime.relu_unstable_mask(relu_idx).ok_or_else(|| {
                        NyError::InternalError(format!(
                            "missing DAG ReLU unstable mask for index {relu_idx}"
                        ))
                    })?;
                    for (i, val) in perturbations[relu_idx].iter_mut().enumerate() {
                        *val = if mask[i] {
                            if rng.random_bool(0.5) {
                                1.0
                            } else {
                                -1.0
                            }
                        } else {
                            0.0
                        };
                    }
                }

                // Refill bilinear perturbations in place (#2297, #3287).
                for pert in bilinear_perts.values_mut() {
                    pert.mapv_inplace(|_| {
                        if rng.random_bool(0.5) {
                            1.0_f32
                        } else {
                            -1.0_f32
                        }
                    });
                }
                // Refill MulBinary perturbations in place (#2297, #3439 Phase 3).
                for pert in mul_binary_perts.values_mut() {
                    pert.mapv_inplace(|_| {
                        if rng.random_bool(0.5) {
                            1.0_f32
                        } else {
                            -1.0_f32
                        }
                    });
                }

                runtime.restore_graph(&original_graph);
                runtime.apply_relu_perturbations(&perturbations, eps)?;
                // Bilinear +eps (#3287).
                for (name, alpha) in bilinear_alphas.iter_mut() {
                    let orig = &original_bilinear[name];
                    let pert = &bilinear_perts[name];
                    ndarray::Zip::from(alpha.view_mut())
                        .and(orig.view())
                        .and(pert.view())
                        .for_each(|a, &o, &p| {
                            *a = (o + eps * p).clamp(0.0, 1.0);
                        });
                }
                // MulBinary +eps (#3439 Phase 3).
                for (name, alpha) in mul_binary_alphas.iter_mut() {
                    let orig = &original_mul_binary[name];
                    let pert = &mul_binary_perts[name];
                    ndarray::Zip::from(alpha.view_mut())
                        .and(orig.view())
                        .and(pert.view())
                        .for_each(|a, &o, &p| {
                            *a = (o + eps * p).clamp(0.0, 1.0);
                        });
                }
                scratch.iter_mut().for_each(|g| g.fill(0.0));
                scratch_upper.iter_mut().for_each(|g| g.fill(0.0));
                // Propagate CROWN errors instead of swallowing them (#2065, #1941).
                let (bl_plus, mb_plus) = alpha_refs(ctx, bilinear_alphas, mul_binary_alphas);
                let bounds_plus = self.dag_alpha_backward_pass_with_engine(
                    ctx.input,
                    node_bounds,
                    ctx.exec_order,
                    ctx.output_dim,
                    ctx.input_dim,
                    runtime.relu_name_to_idx(),
                    runtime.graph(),
                    runtime.invprop(),
                    &mut scratch,
                    &mut scratch_upper,
                    ctx.engine,
                    bl_plus,
                    mb_plus,
                    ctx.config.deadline,
                )?;
                let lower_plus: f32 = finite_lower_sum(bounds_plus.lower());

                runtime.restore_graph(&original_graph);
                runtime.apply_relu_perturbations(&perturbations, -eps)?;
                // Bilinear -eps (#3287).
                for (name, alpha) in bilinear_alphas.iter_mut() {
                    let orig = &original_bilinear[name];
                    let pert = &bilinear_perts[name];
                    ndarray::Zip::from(alpha.view_mut())
                        .and(orig.view())
                        .and(pert.view())
                        .for_each(|a, &o, &p| {
                            *a = (o - eps * p).clamp(0.0, 1.0);
                        });
                }
                // MulBinary -eps (#3439 Phase 3).
                for (name, alpha) in mul_binary_alphas.iter_mut() {
                    let orig = &original_mul_binary[name];
                    let pert = &mul_binary_perts[name];
                    ndarray::Zip::from(alpha.view_mut())
                        .and(orig.view())
                        .and(pert.view())
                        .for_each(|a, &o, &p| {
                            *a = (o - eps * p).clamp(0.0, 1.0);
                        });
                }
                scratch.iter_mut().for_each(|g| g.fill(0.0));
                scratch_upper.iter_mut().for_each(|g| g.fill(0.0));
                // Propagate CROWN errors instead of swallowing them (#2065, #1941).
                let (bl_minus, mb_minus) = alpha_refs(ctx, bilinear_alphas, mul_binary_alphas);
                let bounds_minus = self.dag_alpha_backward_pass_with_engine(
                    ctx.input,
                    node_bounds,
                    ctx.exec_order,
                    ctx.output_dim,
                    ctx.input_dim,
                    runtime.relu_name_to_idx(),
                    runtime.graph(),
                    runtime.invprop(),
                    &mut scratch,
                    &mut scratch_upper,
                    ctx.engine,
                    bl_minus,
                    mb_minus,
                    ctx.config.deadline,
                )?;
                let lower_minus: f32 = finite_lower_sum(bounds_minus.lower());

                let diff = lower_plus - lower_minus;
                for relu_idx in 0..num_relus {
                    let mask = runtime.relu_unstable_mask(relu_idx).ok_or_else(|| {
                        NyError::InternalError(format!(
                            "missing DAG ReLU unstable mask for index {relu_idx}"
                        ))
                    })?;
                    for i in 0..perturbations[relu_idx].len() {
                        if mask[i] && perturbations[relu_idx][i].abs() > 0.5 {
                            avg_grads[relu_idx][i] +=
                                diff / (2.0 * eps * perturbations[relu_idx][i]);
                        }
                    }
                }
                // Bilinear SPSA gradient accumulation (#3287).
                for (name, grad) in bilinear_grads.iter_mut() {
                    if let Some(pert) = bilinear_perts.get(name) {
                        ndarray::Zip::from(grad.view_mut())
                            .and(pert.view())
                            .for_each(|g, &p| {
                                if p.abs() > 0.5 {
                                    *g += diff / (2.0 * eps * p);
                                }
                            });
                    }
                }
                // MulBinary SPSA gradient accumulation (#3439 Phase 3).
                for (name, grad) in mul_binary_grads.iter_mut() {
                    if let Some(pert) = mul_binary_perts.get(name) {
                        ndarray::Zip::from(grad.view_mut())
                            .and(pert.view())
                            .for_each(|g, &p| {
                                if p.abs() > 0.5 {
                                    *g += diff / (2.0 * eps * p);
                                }
                            });
                    }
                }
            }
            Ok(())
        })();

        // Always restore alpha state, including early error
        // returns from `?` inside the closure (#2554, #3393).
        runtime.restore_graph(&original_graph);
        // Restore bilinear alphas from snapshot (#3287).
        *bilinear_alphas = original_bilinear;
        // Restore MulBinary alphas (#3439 Phase 3).
        *mul_binary_alphas = original_mul_binary;
        result?;

        let num_samples = ctx.config.spsa_samples.max(1) as f32;
        for grad in &mut avg_grads {
            *grad /= num_samples;
        }
        // Average bilinear gradients (#3287).
        for grad in bilinear_grads.values_mut() {
            *grad /= num_samples;
        }
        // Average MulBinary gradients (#3439 Phase 3).
        for grad in mul_binary_grads.values_mut() {
            *grad /= num_samples;
        }

        Ok(avg_grads)
    }
}
