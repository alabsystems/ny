// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SPSA supplements for non-ReLU node types in DAG α-CROWN.
//!
//! These supplement non-SPSA gradient methods (AnalyticChain, FD, Analytic)
//! which only compute ReLU gradients — bilinear, MulBinary, S-shaped, Sqrt and
//! reciprocal alphas receive zero gradients without these perturbation-based
//! supplements.
//!
//! (Until #orphans-lies-3 the bilinear half of that sentence was aspirational:
//! there was no BilinearCrown supplement, so on every non-SPSA method — the
//! default `AnalyticChain` included — BilinearCrown alphas were initialized to
//! ones, Adam-stepped with an all-zero gradient, and therefore never moved.)
//!
//! All supplements follow the same pattern:
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
        bilinear_alphas: &mut HashMap<String, Array4<f32>>,
        bilinear_grads: &mut HashMap<String, Array4<f32>>,
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
        // BilinearCrown FIRST, so the remaining supplements read the same
        // (restored, unperturbed) bilinear alphas they read before this arm
        // existed — the added supplement changes only the bilinear gradient.
        if has_bilinear && !matches!(config.gradient_method, GradientMethod::Spsa) {
            self.compute_bilinear_supplement(
                input,
                node_bounds,
                exec_order,
                output_dim,
                input_dim,
                engine,
                runtime,
                gradients,
                bilinear_alphas,
                bilinear_grads,
                mul_binary_alphas,
                has_mul_binary,
                eps,
                iter,
                config.deadline,
            )?;
        }

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

    /// BilinearCrown SPSA supplement (#orphans-lies-3).
    ///
    /// Exactly the MulBinary supplement's shape, for the 4-face McCormick alphas
    /// `[4, m, n, k]` of `BilinearCrown` (attention `Q@K^T`).
    ///
    /// **Why this exists.** This module's own doc has always claimed it supplies
    /// bilinear gradients, but before this arm it did not: `compute_dag_gradients`
    /// fills `bilinear_grads` only under `GradientMethod::Spsa`
    /// (`gradients/spsa.rs:240`); the `AnalyticChain` (DEFAULT), `Analytic` and
    /// `FiniteDifferences` arms leave the map at the zeros it was allocated with
    /// (`gradients/mod.rs:411`). `update_all_alphas` then ran Adam on an all-zero
    /// gradient, which moves nothing (m=v=0 ⇒ step 0), so every BilinearCrown
    /// alpha stayed frozen at its `Array4::ones` initialization for the whole
    /// optimization. The alphas were initialized, threaded into
    /// `DispatchContext::bilinear_alphas`, and consumed by
    /// `propagate_linear_binary_with_alpha` — but never *optimized*.
    ///
    /// **Soundness.** This routine only reads bounds: it perturbs alpha, takes two
    /// backward passes, forms a finite difference, and restores the snapshot
    /// unconditionally (including on the error path). Every alpha it writes goes
    /// through [`clamp_alpha_to_envelope_domain`], the same contracted write site
    /// the Adam updater uses, so alpha stays in `[0, 1]`. For any `r ∈ [0, 1]`
    /// `interpolated_mccormick` returns a convex combination of two valid
    /// McCormick supporting planes over the same input box, with the plane
    /// constant certified outward by `certify_plane_constant` — hence valid for
    /// every alpha this can produce, and the bound cannot become unsound for any
    /// gradient value, finite or not.
    ///
    /// Tightness is likewise protected but by a weaker mechanism, so state it
    /// exactly: the loop maintains `best_lower`/`best_upper` as the ELEMENTWISE
    /// best over the pre-loop CROWN baseline and every returnable iterate (both
    /// arms of the `should_save_best` branch call `update_elementwise_best_bounds`),
    /// so an alpha step that makes an iterate worse cannot displace a better
    /// earlier one. This is not a proof that the returned bound is never worse
    /// than the frozen-alpha bound in every configuration — it is the same
    /// retention every other alpha supplement already relies on.
    ///
    /// A failure here is swallowed (logged at debug) exactly as in the MulBinary
    /// sibling: a missing supplement degrades to the previous zero gradient, which
    /// is the pre-existing behavior, never a weaker bound.
    #[allow(clippy::too_many_arguments)]
    fn compute_bilinear_supplement(
        &self,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        engine: Option<&dyn GemmEngine>,
        runtime: &mut DagAlphaRuntimeState,
        gradients: &[Array1<f32>],
        bilinear_alphas: &mut HashMap<String, Array4<f32>>,
        bilinear_grads: &mut HashMap<String, Array4<f32>>,
        mul_binary_alphas: &HashMap<String, Array2<f32>>,
        has_mul_binary: bool,
        eps: f32,
        iter: usize,
        deadline: Option<std::time::Instant>,
    ) -> Result<()> {
        use rand::RngExt;
        let mut rng = crate::random::rng();

        let original_bilinear: HashMap<String, Array4<f32>> = bilinear_alphas.clone();

        let mut scratch: Vec<Array1<f32>> =
            gradients.iter().map(|g| Array1::zeros(g.len())).collect();
        let mut scratch_upper: Vec<Array1<f32>> =
            gradients.iter().map(|g| Array1::zeros(g.len())).collect();

        // Single-sample SPSA is sufficient — Adam smooths noise across iterations.
        let result = (|| -> Result<()> {
            let bl_perts: HashMap<String, Array4<f32>> = bilinear_alphas
                .iter()
                .map(|(name, alpha)| {
                    let pert = Array4::from_shape_fn(alpha.raw_dim(), |_| {
                        if rng.random_bool(0.5) {
                            1.0_f32
                        } else {
                            -1.0_f32
                        }
                    });
                    (name.clone(), pert)
                })
                .collect();

            let mb_ref = if has_mul_binary {
                Some(mul_binary_alphas)
            } else {
                None
            };

            // +eps bilinear only (ReLU and MulBinary alphas unchanged).
            for (name, alpha) in bilinear_alphas.iter_mut() {
                let orig = &original_bilinear[name];
                let pert = &bl_perts[name];
                ndarray::Zip::from(alpha.view_mut())
                    .and(orig.view())
                    .and(pert.view())
                    .for_each(|a, &o, &p| *a = clamp_alpha_to_envelope_domain(o + eps * p));
            }
            scratch.iter_mut().for_each(|g| g.fill(0.0));
            scratch_upper.iter_mut().for_each(|g| g.fill(0.0));
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
                Some(&*bilinear_alphas),
                mb_ref,
                deadline,
            )?;
            let lower_plus = finite_lower_sum(bounds_plus.lower());

            // -eps bilinear only.
            for (name, alpha) in bilinear_alphas.iter_mut() {
                let orig = &original_bilinear[name];
                let pert = &bl_perts[name];
                ndarray::Zip::from(alpha.view_mut())
                    .and(orig.view())
                    .and(pert.view())
                    .for_each(|a, &o, &p| *a = clamp_alpha_to_envelope_domain(o - eps * p));
            }
            scratch.iter_mut().for_each(|g| g.fill(0.0));
            scratch_upper.iter_mut().for_each(|g| g.fill(0.0));
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
                Some(&*bilinear_alphas),
                mb_ref,
                deadline,
            )?;
            let lower_minus = finite_lower_sum(bounds_minus.lower());

            let diff = lower_plus - lower_minus;
            for (name, grad) in bilinear_grads.iter_mut() {
                if let Some(pert) = bl_perts.get(name) {
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

        // Always restore bilinear alphas, including the error path.
        *bilinear_alphas = original_bilinear;
        if let Err(e) = result {
            debug!(
                "DAG α-CROWN iter {}: BilinearCrown SPSA supplement failed: {}",
                iter, e
            );
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

/// Before/after measurement for the BilinearCrown SPSA supplement (#orphans-lies-3).
///
/// Isolation trick: the fixture's two ReLUs are deliberately **stable** (their
/// pre-activations are strictly positive over the whole input box), so
/// `update_alphas_adam` — which writes only where `unstable_mask[i]` — cannot
/// move a single ReLU alpha. `BilinearCrown` is then the only relaxation in the
/// graph whose alpha can change, while its non-empty `relu_nodes` still keeps
/// `propagate_dag`'s "no optimizable activation nodes" gate from short-circuiting
/// to the batched route. Any width change across iteration counts is therefore
/// attributable to the bilinear alphas and nothing else.
#[cfg(test)]
mod bilinear_supplement_measurement {
    use crate::bounds::{AlphaCrownConfig, GradientMethod};
    use crate::layers::binary_ops::BilinearCrownLayer;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    const LOW: f32 = 0.2;
    const HIGH: f32 = 2.4;

    /// `_input -> {Linear+ReLU (Q), Linear+ReLU (K)} -> BilinearCrown(Q @ K^T)`.
    ///
    /// All projection weights and biases are positive and the input box is
    /// `[0.5, 1.5]^4`, so both ReLU pre-activations are strictly positive: the
    /// ReLUs are stable and contribute no movable alpha.
    fn stable_relu_bilinear_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();

        let proj_q = LinearLayer::new(
            arr2(&[[0.8_f32, 0.3], [0.4, 0.9]]),
            Some(arr1(&[0.2_f32, 0.1])),
        )
        .expect("q proj");
        graph.add_node(GraphNode::from_input("proj_q", Layer::Linear(proj_q)));
        graph.add_node(GraphNode::new(
            "relu_q",
            Layer::ReLU(ReLULayer),
            vec!["proj_q".to_string()],
        ));

        let proj_k = LinearLayer::new(
            arr2(&[[0.5_f32, 0.7], [0.6, 0.4]]),
            Some(arr1(&[0.1_f32, 0.15])),
        )
        .expect("k proj");
        graph.add_node(GraphNode::from_input("proj_k", Layer::Linear(proj_k)));
        graph.add_node(GraphNode::new(
            "relu_k",
            Layer::ReLU(ReLULayer),
            vec!["proj_k".to_string()],
        ));

        graph.add_node(GraphNode::binary(
            "qk",
            Layer::BilinearCrown(BilinearCrownLayer::new(true, Some(1.0 / 2.0_f32.sqrt()))),
            "relu_q",
            "relu_k",
        ));

        // MIXED-SIGN readout. Without it the fixture is degenerate for alpha:
        // over a strictly positive input box with an all-positive output seed,
        // the McCormick lower plane's concrete minimum is `x_l * y_l` at r=1 and
        // strictly smaller for every r < 1, so alpha=1 is provably optimal and no
        // optimizer — engaging or not — can improve on the initialization. The
        // mixed-sign projection puts negative coefficients on some outputs, which
        // is what makes the r_l_neg / r_u_neg faces (slices 1 and 3 of the
        // `[4, m, n, k]` layout) load-bearing.
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0_f32, -1.2], [-0.7, 1.3]]), None).expect("readout"),
            ),
            vec!["qk".to_string()],
        ));
        graph.set_output("out");
        graph
    }

    fn input_box() -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[2, 2]), LOW),
            ArrayD::from_elem(IxDyn(&[2, 2]), HIGH),
        )
        .expect("input box")
    }

    fn total_width(bounds: &BoundedTensor) -> f32 {
        bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .map(|(l, u)| {
                assert!(
                    u - l >= -1e-6 && (u - l).is_finite(),
                    "bad width l={l} u={u}"
                );
                u - l
            })
            .sum()
    }

    fn run(iterations: usize) -> BoundedTensor {
        let config = AlphaCrownConfig {
            iterations,
            adaptive_skip: false,
            gradient_method: GradientMethod::AnalyticChain,
            ..AlphaCrownConfig::default()
        };
        stable_relu_bilinear_graph()
            .propagate_alpha_crown_with_config(&input_box(), &config)
            .expect("alpha-CROWN")
    }

    /// Every point of the input box must land inside the reported enclosure.
    fn assert_encloses_samples(bounds: &BoundedTensor) {
        let graph = stable_relu_bilinear_graph();
        let levels = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
        for sample in 0..levels.len().pow(4) {
            let values: Vec<f32> = (0..4)
                .map(|dim| {
                    let t = levels[(sample / levels.len().pow(dim as u32)) % levels.len()];
                    LOW + t * (HIGH - LOW)
                })
                .collect();
            let point = BoundedTensor::concrete(
                ArrayD::from_shape_vec(IxDyn(&[2, 2]), values).expect("point"),
            )
            .expect("concrete point");
            let exact = graph.propagate_ibp(&point).expect("point forward");
            for (idx, ((&value, &lower), &upper)) in exact
                .lower()
                .iter()
                .zip(bounds.lower().iter())
                .zip(bounds.upper().iter())
                .enumerate()
            {
                assert!(
                    value >= lower - 1e-3 && value <= upper + 1e-3,
                    "sample {sample} output[{idx}] {value} escapes [{lower}, {upper}]"
                );
            }
        }
    }

    /// The supplement makes BilinearCrown alphas actually move under the DEFAULT
    /// gradient method.
    ///
    /// Before the supplement existed, `bilinear_grads` stayed at its allocated
    /// zeros under `AnalyticChain`, Adam's step was identically zero, and the
    /// alphas were pinned at `Array4::ones` — so this graph's bound was
    /// **bit-identical at every iteration count** and this assertion failed.
    ///
    /// MEASURED on this fixture (`iterations` 1 vs 60, `AnalyticChain`, total
    /// output width), by toggling only the `compute_bilinear_supplement` call:
    ///
    /// | | iterations=1 | iterations=60 |
    /// |---|---|---|
    /// | without the supplement | 80.965271 | 80.965271 (0.00%) |
    /// | with the supplement    | 80.965271 | 74.289787 (8.24%) |
    ///
    /// The alpha trajectory over the same run goes `1.000 -> 0.944 -> 0.889 ->
    /// 0.835 -> ...` (mean over the `[4, m, n, k]` array), converging near 0.833
    /// — i.e. the optimum here is a genuine interior interpolation between the
    /// L1/U1 and L2/U2 McCormick faces, not the `ones` initialization.
    #[ntest::timeout(120000)]
    #[test]
    fn bilinear_alphas_move_under_the_default_gradient_method() {
        let one = run(1);
        let many = run(60);
        let (w_one, w_many) = (total_width(&one), total_width(&many));
        println!(
            "BilinearCrown alpha (AnalyticChain): iterations=1 width={w_one:.6}  \
             iterations=60 width={w_many:.6}  improvement={:.2}%",
            (1.0 - w_many / w_one) * 100.0
        );

        // Never looser: the loop keeps `update_elementwise_best_bounds` across
        // iterates, and the frozen alpha=1 bound is iteration 0's.
        assert!(
            w_many <= w_one + 1e-4,
            "more iterations must never widen: {w_many:.6} > {w_one:.6}"
        );
        // Strictly tighter: this is the assertion that fails without the
        // BilinearCrown supplement, because the alphas cannot move. Measured
        // margin on this fixture is ~6.68 absolute (8.24%); the 1.0 threshold
        // leaves room for SPSA sampling noise without admitting a frozen alpha,
        // whose delta is exactly 0.
        assert!(
            w_many < w_one - 1.0,
            "BilinearCrown alphas did not move under AnalyticChain \
             ({w_many:.6} vs {w_one:.6}) — the SPSA supplement is not engaging"
        );
    }

    /// The tighter bound is still an enclosure, and still no looser than IBP.
    #[ntest::timeout(120000)]
    #[test]
    fn optimized_bilinear_bound_stays_sound_and_beats_ibp() {
        let bounds = run(60);
        assert_encloses_samples(&bounds);
        let ibp = stable_relu_bilinear_graph()
            .propagate_ibp(&input_box())
            .expect("IBP");
        assert_encloses_samples(&ibp);
        assert!(
            total_width(&bounds) <= total_width(&ibp) + 1e-4,
            "alpha-CROWN width {:.6} must not exceed IBP {:.6}",
            total_width(&bounds),
            total_width(&ibp)
        );
    }
}
