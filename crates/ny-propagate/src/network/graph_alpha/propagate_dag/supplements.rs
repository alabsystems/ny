// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SPSA supplements for non-ReLU node types in DAG α-CROWN.
//!
//! These supplement non-SPSA gradient methods (AnalyticChain, FD, Analytic)
//! which only compute ReLU gradients — bilinear, S-shaped, and Sqrt alphas
//! receive zero gradients without these perturbation-based supplements.
//!
//! All three supplements follow the same pattern:
//! 1. Snapshot alpha state
//! 2. For each sample: generate Bernoulli ±1 perturbations, apply +eps, backward pass,
//!    apply -eps, backward pass, accumulate gradient
//! 3. Restore alpha state
//! 4. Average over samples

use crate::bounds::alpha_reciprocal::ReciprocalGradients;
use crate::bounds::{AlphaCrownConfig, GradientMethod, MonotoneSShapedGradients, SqrtGradients};

use ndarray::{Array1, Array2, Array4};
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use std::collections::{BTreeMap, HashMap};
use tracing::debug;

use super::super::runtime_state::DagAlphaRuntimeState;
use crate::network::alpha_crown_loop::finite_lower_sum;
use crate::network::core::GraphNetwork;
// #alpha-envelope-domain: the finite-difference probes write alpha too, so they
// share the single contracted write site rather than re-implementing its clamp.
use super::alpha_update::clamp_alpha_to_envelope_domain;

impl GraphNetwork {
    /// Compute SPSA supplements for all non-ReLU node types.
    ///
    /// Called after the primary gradient computation when using non-SPSA gradient
    /// methods (AnalyticChain, FD, Analytic) that only produce ReLU gradients.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_spsa_supplements(
        &self,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        runtime: &mut DagAlphaRuntimeState,
        gradients: &[Array1<f32>],
        bilinear_alphas: &HashMap<String, Array4<f32>>,
        mul_binary_alphas: &mut HashMap<String, Array2<f32>>,
        mul_binary_grads: &mut HashMap<String, Array2<f32>>,
        s_shaped_grads: &mut BTreeMap<String, MonotoneSShapedGradients>,
        sqrt_grads: &mut BTreeMap<String, SqrtGradients>,
        reciprocal_grads: &mut BTreeMap<String, ReciprocalGradients>,
        has_bilinear: bool,
        has_mul_binary: bool,
        has_s_shaped: bool,
        has_sqrt: bool,
        has_reciprocal: bool,
        eps: f32,
        iter: usize,
    ) -> Result<()> {
        if has_mul_binary && !matches!(config.gradient_method, GradientMethod::Spsa) {
            self.compute_mul_binary_supplement(
                input,
                node_bounds,
                exec_order,
                output_dim,
                input_dim,
                engine,
                runtime,
                gradients,
                bilinear_alphas,
                mul_binary_alphas,
                mul_binary_grads,
                has_bilinear,
                eps,
                iter,
                config.deadline,
            )?;
        }

        if has_s_shaped {
            self.compute_s_shaped_supplement(
                input,
                node_bounds,
                exec_order,
                output_dim,
                input_dim,
                config,
                engine,
                runtime,
                gradients,
                bilinear_alphas,
                mul_binary_alphas,
                s_shaped_grads,
                has_bilinear,
                has_mul_binary,
                eps,
                iter,
            )?;
        }

        if has_sqrt {
            self.compute_sqrt_supplement(
                input,
                node_bounds,
                exec_order,
                output_dim,
                input_dim,
                config,
                engine,
                runtime,
                gradients,
                bilinear_alphas,
                mul_binary_alphas,
                sqrt_grads,
                has_bilinear,
                has_mul_binary,
                eps,
                iter,
            )?;
        }

        if has_reciprocal {
            self.compute_reciprocal_supplement(
                input,
                node_bounds,
                exec_order,
                output_dim,
                input_dim,
                config,
                engine,
                runtime,
                gradients,
                bilinear_alphas,
                mul_binary_alphas,
                reciprocal_grads,
                has_bilinear,
                has_mul_binary,
                eps,
                iter,
            )?;
        }

        Ok(())
    }

    /// MulBinary SPSA supplement (#3439 Phase 3).
    ///
    /// Targeted SPSA for MulBinary alphas when using non-SPSA methods. AnalyticChain/
    /// Analytic/FD compute only ReLU gradients — MulBinary alphas receive zero gradients
    /// (dead). This adds perturbation-based gradients for MulBinary alphas while keeping
    /// ReLU alphas fixed.
    #[allow(clippy::too_many_arguments)]
    fn compute_mul_binary_supplement(
        &self,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        engine: Option<&dyn GemmEngine>,
        runtime: &mut DagAlphaRuntimeState,
        gradients: &[Array1<f32>],
        bilinear_alphas: &HashMap<String, Array4<f32>>,
        mul_binary_alphas: &mut HashMap<String, Array2<f32>>,
        mul_binary_grads: &mut HashMap<String, Array2<f32>>,
        has_bilinear: bool,
        eps: f32,
        iter: usize,
        deadline: Option<std::time::Instant>,
    ) -> Result<()> {
        use rand::RngExt;
        let mut rng = crate::random::rng();

        let original_mul_binary: HashMap<String, Array2<f32>> = mul_binary_alphas.clone();

        let mut scratch_mb: Vec<Array1<f32>> =
            gradients.iter().map(|g| Array1::zeros(g.len())).collect();
        let mut scratch_mb_upper: Vec<Array1<f32>> =
            gradients.iter().map(|g| Array1::zeros(g.len())).collect();

        // Single-sample SPSA is sufficient — Adam smooths noise across iterations.
        let result = (|| -> Result<()> {
            let mb_perts: HashMap<String, Array2<f32>> = mul_binary_alphas
                .iter()
                .map(|(name, alpha)| {
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

            // +eps MulBinary only (ReLU alphas unchanged).
            for (name, alpha) in mul_binary_alphas.iter_mut() {
                let orig = &original_mul_binary[name];
                let pert = &mb_perts[name];
                ndarray::Zip::from(alpha.view_mut())
                    .and(orig.view())
                    .and(pert.view())
                    .for_each(|a, &o, &p| *a = clamp_alpha_to_envelope_domain(o + eps * p));
            }
            scratch_mb.iter_mut().for_each(|g| g.fill(0.0));
            scratch_mb_upper.iter_mut().for_each(|g| g.fill(0.0));
            let bl_ref = if has_bilinear {
                Some(bilinear_alphas)
            } else {
                None
            };
            let mb_plus = Some(&*mul_binary_alphas as &HashMap<String, Array2<f32>>);
            let bounds_plus = self.dag_alpha_backward_pass_with_engine(
                input,
                node_bounds,
                exec_order,
                output_dim,
                input_dim,
                runtime.relu_name_to_idx(),
                runtime.graph(),
                runtime.invprop(),
                &mut scratch_mb,
                &mut scratch_mb_upper,
                engine,
                bl_ref,
                mb_plus,
                deadline,
            )?;
            let lower_plus = finite_lower_sum(bounds_plus.lower());

            // -eps MulBinary only.
            for (name, alpha) in mul_binary_alphas.iter_mut() {
                let orig = &original_mul_binary[name];
                let pert = &mb_perts[name];
                ndarray::Zip::from(alpha.view_mut())
                    .and(orig.view())
                    .and(pert.view())
                    .for_each(|a, &o, &p| *a = clamp_alpha_to_envelope_domain(o - eps * p));
            }
            scratch_mb.iter_mut().for_each(|g| g.fill(0.0));
            scratch_mb_upper.iter_mut().for_each(|g| g.fill(0.0));
            let mb_minus = Some(&*mul_binary_alphas as &HashMap<String, Array2<f32>>);
            let bounds_minus = self.dag_alpha_backward_pass_with_engine(
                input,
                node_bounds,
                exec_order,
                output_dim,
                input_dim,
                runtime.relu_name_to_idx(),
                runtime.graph(),
                runtime.invprop(),
                &mut scratch_mb,
                &mut scratch_mb_upper,
                engine,
                bl_ref,
                mb_minus,
                deadline,
            )?;
            let lower_minus = finite_lower_sum(bounds_minus.lower());

            let diff = lower_plus - lower_minus;
            for (name, grad) in mul_binary_grads.iter_mut() {
                if let Some(pert) = mb_perts.get(name) {
                    ndarray::Zip::from(grad.view_mut())
                        .and(pert.view())
                        .for_each(|g, &p| {
                            if p.abs() > 0.5 {
                                *g += diff / (2.0 * eps * p);
                            }
                        });
                }
            }
            Ok(())
        })();

        // Always restore MulBinary alphas.
        *mul_binary_alphas = original_mul_binary;
        if let Err(e) = result {
            debug!(
                "DAG α-CROWN iter {}: MulBinary SPSA supplement failed: {}",
                iter, e
            );
        }
        Ok(())
    }

    /// Monotone S-shaped (Sigmoid/Tanh) SPSA supplement (#3619 Packet A).
    ///
    /// Tangent-point parameters do not yet have a chain-rule gradient path.
    /// Uses perturbation-based SPSA loop with `config.spsa_samples` samples.
    #[allow(clippy::too_many_arguments)]
    fn compute_s_shaped_supplement(
        &self,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        runtime: &mut DagAlphaRuntimeState,
        gradients: &[Array1<f32>],
        bilinear_alphas: &HashMap<String, Array4<f32>>,
        mul_binary_alphas: &HashMap<String, Array2<f32>>,
        s_shaped_grads: &mut BTreeMap<String, MonotoneSShapedGradients>,
        has_bilinear: bool,
        has_mul_binary: bool,
        eps: f32,
        iter: usize,
    ) -> Result<()> {
        let mut rng = crate::random::rng();
        let original_graph = runtime.snapshot_graph();
        let spsa_samples = config.spsa_samples.max(1);
        let mut scratch_s: Vec<Array1<f32>> =
            gradients.iter().map(|g| Array1::zeros(g.len())).collect();
        let mut scratch_s_upper: Vec<Array1<f32>> =
            gradients.iter().map(|g| Array1::zeros(g.len())).collect();

        let result = (|| -> Result<()> {
            for _sample in 0..spsa_samples {
                let perturbations: BTreeMap<String, MonotoneSShapedGradients> = runtime
                    .graph()
                    .monotone_alpha_names()
                    .map(|name| {
                        let alpha =
                            runtime
                                .graph()
                                .monotone_s_shaped_alpha(name)
                                .ok_or_else(|| {
                                    ny_core::NyError::InternalError(
                                        "monotone alpha iterator must point to existing state"
                                            .into(),
                                    )
                                })?;
                        Ok((name.clone(), alpha.spsa_perturbations(&mut rng)))
                    })
                    .collect::<Result<_>>()?;

                runtime.restore_graph(&original_graph);
                runtime.apply_monotone_perturbations(&perturbations, eps)?;
                scratch_s.iter_mut().for_each(|g| g.fill(0.0));
                scratch_s_upper.iter_mut().for_each(|g| g.fill(0.0));
                let bl_plus = if has_bilinear {
                    Some(bilinear_alphas)
                } else {
                    None
                };
                let mb_plus = if has_mul_binary {
                    Some(mul_binary_alphas)
                } else {
                    None
                };
                let bounds_plus = self.dag_alpha_backward_pass_with_engine(
                    input,
                    node_bounds,
                    exec_order,
                    output_dim,
                    input_dim,
                    runtime.relu_name_to_idx(),
                    runtime.graph(),
                    runtime.invprop(),
                    &mut scratch_s,
                    &mut scratch_s_upper,
                    engine,
                    bl_plus,
                    mb_plus,
                    config.deadline,
                )?;
                let lower_plus = finite_lower_sum(bounds_plus.lower());

                runtime.restore_graph(&original_graph);
                runtime.apply_monotone_perturbations(&perturbations, -eps)?;
                scratch_s.iter_mut().for_each(|g| g.fill(0.0));
                scratch_s_upper.iter_mut().for_each(|g| g.fill(0.0));
                let bl_minus = if has_bilinear {
                    Some(bilinear_alphas)
                } else {
                    None
                };
                let mb_minus = if has_mul_binary {
                    Some(mul_binary_alphas)
                } else {
                    None
                };
                let bounds_minus = self.dag_alpha_backward_pass_with_engine(
                    input,
                    node_bounds,
                    exec_order,
                    output_dim,
                    input_dim,
                    runtime.relu_name_to_idx(),
                    runtime.graph(),
                    runtime.invprop(),
                    &mut scratch_s,
                    &mut scratch_s_upper,
                    engine,
                    bl_minus,
                    mb_minus,
                    config.deadline,
                )?;
                let lower_minus = finite_lower_sum(bounds_minus.lower());
                let diff = lower_plus - lower_minus;
                let scale = diff / (2.0 * eps);

                for (name, perturbation) in &perturbations {
                    let Some(grad) = s_shaped_grads.get_mut(name) else {
                        continue;
                    };
                    for i in 0..grad.tp_pos.lower_path.len() {
                        if perturbation.tp_pos.lower_path[i].abs() > 0.5 {
                            grad.tp_pos.lower_path[i] += scale / perturbation.tp_pos.lower_path[i];
                        }
                        if perturbation.tp_pos.upper_path[i].abs() > 0.5 {
                            grad.tp_pos.upper_path[i] += scale / perturbation.tp_pos.upper_path[i];
                        }
                        if perturbation.tp_neg.lower_path[i].abs() > 0.5 {
                            grad.tp_neg.lower_path[i] += scale / perturbation.tp_neg.lower_path[i];
                        }
                        if perturbation.tp_neg.upper_path[i].abs() > 0.5 {
                            grad.tp_neg.upper_path[i] += scale / perturbation.tp_neg.upper_path[i];
                        }
                        if perturbation.tp_both_lower.lower_path[i].abs() > 0.5 {
                            grad.tp_both_lower.lower_path[i] +=
                                scale / perturbation.tp_both_lower.lower_path[i];
                        }
                        if perturbation.tp_both_lower.upper_path[i].abs() > 0.5 {
                            grad.tp_both_lower.upper_path[i] +=
                                scale / perturbation.tp_both_lower.upper_path[i];
                        }
                        if perturbation.tp_both_upper.lower_path[i].abs() > 0.5 {
                            grad.tp_both_upper.lower_path[i] +=
                                scale / perturbation.tp_both_upper.lower_path[i];
                        }
                        if perturbation.tp_both_upper.upper_path[i].abs() > 0.5 {
                            grad.tp_both_upper.upper_path[i] +=
                                scale / perturbation.tp_both_upper.upper_path[i];
                        }
                    }
                }
            }
            Ok(())
        })();

        runtime.restore_graph(&original_graph);
        if let Err(e) = result {
            debug!(
                "DAG α-CROWN iter {}: monotone S-shaped SPSA supplement failed: {}",
                iter, e
            );
        } else {
            let inv_samples = 1.0 / (spsa_samples as f32);
            for grad in s_shaped_grads.values_mut() {
                grad.scale_in_place(inv_samples);
            }
        }
        Ok(())
    }

    /// Sqrt SPSA supplement.
    ///
    /// Same pattern as S-shaped supplement but for Sqrt tangent-point parameters.
    #[allow(clippy::too_many_arguments)]
    fn compute_sqrt_supplement(
        &self,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        runtime: &mut DagAlphaRuntimeState,
        gradients: &[Array1<f32>],
        bilinear_alphas: &HashMap<String, Array4<f32>>,
        mul_binary_alphas: &HashMap<String, Array2<f32>>,
        sqrt_grads: &mut BTreeMap<String, SqrtGradients>,
        has_bilinear: bool,
        has_mul_binary: bool,
        eps: f32,
        iter: usize,
    ) -> Result<()> {
        let mut rng = crate::random::rng();
        let original_graph = runtime.snapshot_graph();
        let spsa_samples = config.spsa_samples.max(1);
        let mut scratch_q: Vec<Array1<f32>> =
            gradients.iter().map(|g| Array1::zeros(g.len())).collect();
        let mut scratch_q_upper: Vec<Array1<f32>> =
            gradients.iter().map(|g| Array1::zeros(g.len())).collect();

        let result = (|| -> Result<()> {
            for _sample in 0..spsa_samples {
                let perturbations: BTreeMap<String, SqrtGradients> = runtime
                    .graph()
                    .sqrt_alpha_names()
                    .map(|name| {
                        let alpha = runtime.graph().sqrt_alpha(name).ok_or_else(|| {
                            ny_core::NyError::InternalError(
                                "sqrt alpha iterator must point to existing state".into(),
                            )
                        })?;
                        Ok((name.clone(), alpha.spsa_perturbations(&mut rng)))
                    })
                    .collect::<Result<_>>()?;

                runtime.restore_graph(&original_graph);
                runtime.apply_sqrt_perturbations(&perturbations, eps)?;
                scratch_q.iter_mut().for_each(|g| g.fill(0.0));
                scratch_q_upper.iter_mut().for_each(|g| g.fill(0.0));
                let bl_plus = if has_bilinear {
                    Some(bilinear_alphas)
                } else {
                    None
                };
                let mb_plus = if has_mul_binary {
                    Some(mul_binary_alphas)
                } else {
                    None
                };
                let bounds_plus = self.dag_alpha_backward_pass_with_engine(
                    input,
                    node_bounds,
                    exec_order,
                    output_dim,
                    input_dim,
                    runtime.relu_name_to_idx(),
                    runtime.graph(),
                    runtime.invprop(),
                    &mut scratch_q,
                    &mut scratch_q_upper,
                    engine,
                    bl_plus,
                    mb_plus,
                    config.deadline,
                )?;
                let lower_plus = finite_lower_sum(bounds_plus.lower());

                runtime.restore_graph(&original_graph);
                runtime.apply_sqrt_perturbations(&perturbations, -eps)?;
                scratch_q.iter_mut().for_each(|g| g.fill(0.0));
                scratch_q_upper.iter_mut().for_each(|g| g.fill(0.0));
                let bl_minus = if has_bilinear {
                    Some(bilinear_alphas)
                } else {
                    None
                };
                let mb_minus = if has_mul_binary {
                    Some(mul_binary_alphas)
                } else {
                    None
                };
                let bounds_minus = self.dag_alpha_backward_pass_with_engine(
                    input,
                    node_bounds,
                    exec_order,
                    output_dim,
                    input_dim,
                    runtime.relu_name_to_idx(),
                    runtime.graph(),
                    runtime.invprop(),
                    &mut scratch_q,
                    &mut scratch_q_upper,
                    engine,
                    bl_minus,
                    mb_minus,
                    config.deadline,
                )?;
                let lower_minus = finite_lower_sum(bounds_minus.lower());
                let diff = lower_plus - lower_minus;

                for (name, perturbation) in &perturbations {
                    let Some(grad) = sqrt_grads.get_mut(name) else {
                        continue;
                    };
                    for i in 0..grad.lower_path.len() {
                        if perturbation.lower_path[i].abs() > 0.5 {
                            grad.lower_path[i] += diff / (2.0 * eps * perturbation.lower_path[i]);
                        }
                    }
                    for i in 0..grad.upper_path.len() {
                        if perturbation.upper_path[i].abs() > 0.5 {
                            grad.upper_path[i] += diff / (2.0 * eps * perturbation.upper_path[i]);
                        }
                    }
                }
            }
            Ok(())
        })();

        runtime.restore_graph(&original_graph);
        if let Err(e) = result {
            debug!(
                "DAG α-CROWN iter {}: sqrt SPSA supplement failed: {}",
                iter, e
            );
        } else {
            let inv_samples = 1.0 / (spsa_samples as f32);
            for grad in sqrt_grads.values_mut() {
                grad.scale_in_place(inv_samples);
            }
        }
        Ok(())
    }

    /// Reciprocal SPSA supplement (#4399 Slice 2).
    ///
    /// Same pattern as Sqrt supplement but for Reciprocal tangent-point parameters.
    #[allow(clippy::too_many_arguments)]
    fn compute_reciprocal_supplement(
        &self,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        runtime: &mut DagAlphaRuntimeState,
        gradients: &[Array1<f32>],
        bilinear_alphas: &HashMap<String, Array4<f32>>,
        mul_binary_alphas: &HashMap<String, Array2<f32>>,
        reciprocal_grads: &mut BTreeMap<String, ReciprocalGradients>,
        has_bilinear: bool,
        has_mul_binary: bool,
        eps: f32,
        iter: usize,
    ) -> Result<()> {
        let mut rng = crate::random::rng();
        let original_graph = runtime.snapshot_graph();
        let spsa_samples = config.spsa_samples.max(1);
        let mut scratch: Vec<Array1<f32>> =
            gradients.iter().map(|g| Array1::zeros(g.len())).collect();
        let mut scratch_upper: Vec<Array1<f32>> =
            gradients.iter().map(|g| Array1::zeros(g.len())).collect();

        let result = (|| -> Result<()> {
            for _sample in 0..spsa_samples {
                let perturbations: BTreeMap<String, ReciprocalGradients> = runtime
                    .graph()
                    .reciprocal_alpha_names()
                    .map(|name| {
                        let alpha = runtime.graph().reciprocal_alpha(name).ok_or_else(|| {
                            ny_core::NyError::InternalError(
                                "reciprocal alpha iterator must point to existing state".into(),
                            )
                        })?;
                        Ok((name.clone(), alpha.spsa_perturbations(&mut rng)))
                    })
                    .collect::<Result<_>>()?;

                runtime.restore_graph(&original_graph);
                runtime.apply_reciprocal_perturbations(&perturbations, eps)?;
                scratch.iter_mut().for_each(|g| g.fill(0.0));
                scratch_upper.iter_mut().for_each(|g| g.fill(0.0));
                let bl_plus = if has_bilinear {
                    Some(bilinear_alphas)
                } else {
                    None
                };
                let mb_plus = if has_mul_binary {
                    Some(mul_binary_alphas)
                } else {
                    None
                };
                let bounds_plus = self.dag_alpha_backward_pass_with_engine(
                    input,
                    node_bounds,
                    exec_order,
                    output_dim,
                    input_dim,
                    runtime.relu_name_to_idx(),
                    runtime.graph(),
                    runtime.invprop(),
                    &mut scratch,
                    &mut scratch_upper,
                    engine,
                    bl_plus,
                    mb_plus,
                    config.deadline,
                )?;
                let lower_plus = finite_lower_sum(bounds_plus.lower());

                runtime.restore_graph(&original_graph);
                runtime.apply_reciprocal_perturbations(&perturbations, -eps)?;
                scratch.iter_mut().for_each(|g| g.fill(0.0));
                scratch_upper.iter_mut().for_each(|g| g.fill(0.0));
                let bl_minus = if has_bilinear {
                    Some(bilinear_alphas)
                } else {
                    None
                };
                let mb_minus = if has_mul_binary {
                    Some(mul_binary_alphas)
                } else {
                    None
                };
                let bounds_minus = self.dag_alpha_backward_pass_with_engine(
                    input,
                    node_bounds,
                    exec_order,
                    output_dim,
                    input_dim,
                    runtime.relu_name_to_idx(),
                    runtime.graph(),
                    runtime.invprop(),
                    &mut scratch,
                    &mut scratch_upper,
                    engine,
                    bl_minus,
                    mb_minus,
                    config.deadline,
                )?;
                let lower_minus = finite_lower_sum(bounds_minus.lower());
                let diff = lower_plus - lower_minus;

                for (name, perturbation) in &perturbations {
                    let Some(grad) = reciprocal_grads.get_mut(name) else {
                        continue;
                    };
                    for i in 0..grad.lower_path.len() {
                        if perturbation.lower_path[i].abs() > 0.5 {
                            grad.lower_path[i] += diff / (2.0 * eps * perturbation.lower_path[i]);
                        }
                    }
                    for i in 0..grad.upper_path.len() {
                        if perturbation.upper_path[i].abs() > 0.5 {
                            grad.upper_path[i] += diff / (2.0 * eps * perturbation.upper_path[i]);
                        }
                    }
                }
            }
            Ok(())
        })();

        runtime.restore_graph(&original_graph);
        if let Err(e) = result {
            debug!(
                "DAG α-CROWN iter {}: reciprocal SPSA supplement failed: {}",
                iter, e
            );
        } else {
            let inv_samples = 1.0 / (spsa_samples as f32);
            for grad in reciprocal_grads.values_mut() {
                grad.scale_in_place(inv_samples);
            }
        }
        Ok(())
    }
}
