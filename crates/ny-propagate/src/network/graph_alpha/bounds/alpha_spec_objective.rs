// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-disjunct spec-objective alpha optimizer (#4355).
//!
//! Extracted from `alpha.rs` to keep file sizes under 500 lines.

use crate::bounds::{AlphaCrownConfig, GraphAlphaState, Optimizer};
use crate::network::core::GraphNetwork;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::debug;

use crate::network::graph_alpha::spsa::DagSpsaGradients;

/// Dark direct-C objective gate. Only the exact string `"1"` enables it, so
/// unset, `"0"`, and malformed values preserve the legacy optimizer byte path.
fn spec_alpha_direct_enabled_for(value: Option<&str>) -> bool {
    value == Some("1")
}

fn spec_alpha_direct_enabled() -> bool {
    spec_alpha_direct_enabled_for(std::env::var("NY_SPEC_ALPHA_DIRECT").ok().as_deref())
}

/// One internally consistent evaluation from the existing sound GPU direct-C
/// fold: the objective and gradients were produced for the same spec row and
/// the same alpha state in the same backend call.
struct GpuSpecObjectiveEvaluation {
    lower_bound: f32,
    gradients: DagSpsaGradients,
}

impl GpuSpecObjectiveEvaluation {
    /// A direct iteration is usable only when both its sound objective and
    /// every proposed update are finite. Non-finite gradients are not a
    /// soundness risk by themselves, but accepting one would violate the
    /// fail-closed experiment contract and could strand the optimizer on a
    /// state that the legacy fallback would have advanced.
    fn is_finite(&self) -> bool {
        self.lower_bound.is_finite()
            && self
                .gradients
                .relu
                .values()
                .all(|gradient| gradient.iter().all(|value| value.is_finite()))
            && self
                .gradients
                .monotone
                .values()
                .all(|gradient| !gradient.any_non_finite())
            && self
                .gradients
                .sqrt
                .values()
                .all(|gradient| !gradient.any_non_finite())
            && self
                .gradients
                .reciprocal
                .values()
                .all(|gradient| !gradient.any_non_finite())
    }
}

impl GraphNetwork {
    /// Optimize alpha for a single spec-row objective (#4355).
    ///
    /// Takes pre-computed IBP bounds and an initial `GraphAlphaState`, runs the
    /// SPSA optimization loop targeting the lower bound of `spec_row^T output`,
    /// and returns the optimized alpha state.
    ///
    /// Used by `optimize_disjuncts_separately` to produce per-disjunct alpha
    /// states at bootstrap time. Each disjunct gets its own alpha specialized
    /// for proving that specific output constraint.
    ///
    /// Reference: alpha-beta-CROWN `beta_CROWN_solver.py:1098`
    pub(crate) fn optimize_alpha_for_spec_objective(
        &self,
        input: &BoundedTensor,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        initial_alpha: &GraphAlphaState,
        config: &AlphaCrownConfig,
        spec_row: &[f32],
        engine: Option<&dyn ny_core::GemmEngine>,
    ) -> Result<GraphAlphaState> {
        use crate::network::graph_alpha::spsa::spec_guided_lower;

        let output_node = if self.output_node.is_empty() {
            let exec_order = self.exec_order()?;
            exec_order.last().cloned().ok_or_else(|| {
                NyError::InvalidSpec("spec-objective optimization: empty exec_order".to_string())
            })?
        } else {
            self.output_node.clone()
        };

        let mut alpha_state = initial_alpha.clone();
        let relu_nodes: Vec<String> = alpha_state.alphas.keys().cloned().collect();
        let s_shaped_nodes: Vec<String> = alpha_state.monotone_alpha_names().cloned().collect();
        let sqrt_nodes: Vec<String> = alpha_state.sqrt_alpha_names().cloned().collect();

        let mut lr = config.learning_rate;
        let eps = 1e-3;
        let mut best_objective = f32::NEG_INFINITY;
        // #spec-alpha-direct (dark): while the sound GPU direct-C fold remains
        // available, its lower bound is the optimization objective and the
        // best matching alpha state is retained. A single miss/non-finite evaluation
        // permanently returns this invocation to the exact legacy path and
        // discards the direct snapshot. It then resumes today's legacy update
        // path from the current valid alpha state; skipped earlier surrogate
        // calls can still make wall-clock, deadline, and cache history differ.
        let direct_requested = spec_alpha_direct_enabled();
        let mut direct_active = direct_requested;
        let mut best_direct_alpha: Option<GraphAlphaState> = None;
        if crate::beta_gpu_probe_armed() {
            if direct_requested {
                eprintln!(
                    "[spec-obj-opt] iterations={} relus={} od={} direct=true",
                    config.iterations,
                    relu_nodes.len(),
                    spec_row.len()
                );
            } else {
                // Preserve the gate-off diagnostic byte-for-byte.
                eprintln!(
                    "[spec-obj-opt] iterations={} relus={} od={}",
                    config.iterations,
                    relu_nodes.len(),
                    spec_row.len()
                );
            }
        }

        for iter in 0..config.iterations {
            if config.past_deadline() {
                break;
            }

            // In direct mode this is the sole full-network evaluation for a
            // successful iteration: its verdict-safe `lower_bounds[0]` and its
            // alpha gradient come from the same direct-C GPU fold.
            let direct_evaluation = if direct_active {
                self.try_gpu_spec_objective_evaluation(
                    input,
                    ibp_bounds,
                    &alpha_state,
                    &output_node,
                    spec_row,
                    engine,
                    config.deadline,
                )
            } else {
                None
            };

            let gradients = if let Some(evaluation) =
                direct_evaluation.filter(GpuSpecObjectiveEvaluation::is_finite)
            {
                let objective = evaluation.lower_bound;
                if objective > best_objective {
                    best_objective = objective;
                    best_direct_alpha = Some(alpha_state.clone());
                }
                if crate::beta_gpu_probe_armed() {
                    eprintln!(
                        "[spec-obj-opt] iter={iter} source=direct objective={objective:.9} best={best_objective:.9}"
                    );
                }
                if iter == config.iterations - 1 {
                    break;
                }
                evaluation.gradients
            } else {
                if direct_active {
                    // Fail closed for the rest of this optimization. Do not
                    // compare the independent-output surrogate against a
                    // direct-C score: they are different relaxations/objectives.
                    direct_active = false;
                    best_direct_alpha = None;
                    best_objective = f32::NEG_INFINITY;
                    if crate::beta_gpu_probe_armed() {
                        eprintln!(
                            "[spec-obj-opt] iter={iter} source=direct unavailable; legacy fallback"
                        );
                    }
                }

                // Legacy path below is intentionally unchanged: independently
                // bound each output, fold the spec by interval arithmetic, then
                // request the GPU analytic gradient or fall back to SPSA.
                let output_bounds = match self.propagate_crown_to_node_with_alpha(
                    input,
                    &output_node,
                    &std::collections::HashMap::new(),
                    ibp_bounds,
                    &alpha_state,
                    engine,
                    config.deadline,
                ) {
                    Ok(bounds) => bounds,
                    // CpuMemoryExceeded: Conv2d backward memory-cap backstop
                    // (#conv-crown-oom); break to the sound IBP objective fallback.
                    Err(
                        NyError::UnsupportedOp(_)
                        | NyError::UnsupportedConfiguration(_)
                        | NyError::ShapeMismatch { .. }
                        | NyError::CpuMemoryExceeded { .. }
                        | NyError::DeadlineExceeded(_),
                    ) => break,
                    Err(e) => return Err(e),
                };

                // Intersect the raw backward output bound with the always-available IBP output
                // bound before it drives the spec-guided objective. SOUND: both enclose the
                // output node's reachable set, so the per-element intersection (union on
                // disjoint) still encloses it — never looser; on NaN/shape-mismatch (None)
                // keep the CROWN bound unchanged (sound; NaN-guarded downstream). Mirrors the
                // post-loop IBP intersection used elsewhere.
                let output_bounds = match ibp_bounds.get(&output_node) {
                    Some(ibp_out) if ibp_out.shape() == output_bounds.shape() => output_bounds
                        .intersection_per_element(ibp_out)
                        .map(|(t, _)| t)
                        .unwrap_or(output_bounds),
                    _ => output_bounds,
                };

                let objective = spec_guided_lower(&output_bounds, spec_row);
                if objective.is_finite() && objective > best_objective {
                    best_objective = objective;
                }

                if iter == config.iterations - 1 {
                    break;
                }

                // #unsat-keystone step 3b-ii: GPU analytic spec-objective gradient. SPSA needs
                // spsa_samples bound-evals PER gradient (×99 disjuncts ×iterations = the warmup
                // overrun that keeps cifar100 BaB at ~0 domains); the GPU resnet backward gives
                // the per-ReLU analytic gradient in ONE call. Default ON (opt out
                // NY_RESNET_WARMUP_GPU=0); on any miss → the proven SPSA path. Gradient is
                // non-soundness-critical.
                match self.try_gpu_spec_objective_evaluation(
                    input,
                    ibp_bounds,
                    &alpha_state,
                    &output_node,
                    spec_row,
                    engine,
                    config.deadline,
                ) {
                    Some(evaluation) => evaluation.gradients,
                    None => self.compute_spsa_gradients_for_spec_objective(
                        input,
                        ibp_bounds,
                        &alpha_state,
                        &output_node,
                        eps,
                        config.spsa_samples,
                        None,
                        engine,
                        spec_row,
                    )?,
                }
            };

            let adam_params = config.adam_params(lr, iter + 1);
            for relu_name in &relu_nodes {
                if let Some(grad) = gradients.relu.get(relu_name) {
                    if grad.iter().any(|v| !v.is_finite()) {
                        continue;
                    }
                    let neg_grad = -grad;
                    // Channel-only alpha reduction (#4404)
                    let neg_grad = alpha_state.reduce_gradient(relu_name, &neg_grad);
                    match config.optimizer {
                        Optimizer::Adam => {
                            alpha_state.update_adam(relu_name, &neg_grad, &adam_params);
                            alpha_state.update_adam_upper(relu_name, &neg_grad, &adam_params);
                        }
                        Optimizer::Sgd => {
                            alpha_state.update(relu_name, &neg_grad, lr, config.momentum);
                            alpha_state.update_upper(relu_name, &neg_grad, lr, config.momentum);
                        }
                    }
                }
            }
            for node_name in &s_shaped_nodes {
                if let Some(grad) = gradients.monotone.get(node_name) {
                    if grad.any_non_finite() {
                        continue;
                    }
                    let neg_grad = grad.negate();
                    if let Some(alpha) = alpha_state.monotone_s_shaped_alpha_mut(node_name) {
                        match config.optimizer {
                            Optimizer::Adam => alpha.update_adam(&neg_grad, &adam_params),
                            Optimizer::Sgd => alpha.update_sgd(&neg_grad, lr, config.momentum),
                        }
                    }
                }
            }
            super::sqrt_support::update_sqrt_alpha_gradients(
                &sqrt_nodes,
                &gradients.sqrt,
                &mut alpha_state,
                config.optimizer,
                &adam_params,
                lr,
                config.momentum,
                iter,
            );

            lr *= config.lr_decay;
        }

        debug!(
            "Per-disjunct α-CROWN: optimized for spec_row (len={}), best_objective={best_objective:.4}",
            spec_row.len()
        );
        Ok(if direct_active {
            best_direct_alpha.unwrap_or(alpha_state)
        } else {
            alpha_state
        })
    }

    /// GPU analytic gradient of `spec_row^T (lower output)` w.r.t. each ReLU alpha, in ONE
    /// sound GPU resnet backward — the analytic replacement for the SPSA warmup gradient
    /// (#unsat-keystone step 3b-ii). Returns per-neuron (un-reduced) gradients in the
    /// `DagSpsaGradients` shape the optimizer expects (caller reduces per-channel at the
    /// update site). Default ON, opt out `NY_RESNET_WARMUP_GPU=0`; None (no sound GPU /
    /// non-resnet / any mismatch) → SPSA fallback. Non-soundness-critical: any alpha∈[0,1]
    /// is sound, so a convention mismatch yields only suboptimal alpha, never a wrong
    /// verdict.
    fn try_gpu_spec_objective_evaluation(
        &self,
        input: &BoundedTensor,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        output_node: &str,
        spec_row: &[f32],
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
    ) -> Option<GpuSpecObjectiveEvaluation> {
        use crate::network::core::NETWORK_INPUT;
        use ndarray::Array1;
        if !crate::network::graph_alpha::resnet_decompose::resnet_warmup_gpu_enabled() {
            return None;
        }
        let gpu = engine
            .and_then(|e| e.as_gpu_crown_backward())
            .filter(|g| g.provides_sound_gpu_crown())?;
        if !crate::sound_gpu_gate::gpu_crown_backend_honors_deadline(gpu, deadline)
            || deadline.is_some_and(|value| Instant::now() >= value)
        {
            return None;
        }
        let (segments, relu_names, frontier_abs, node_abs) =
            crate::network::graph_alpha::resnet_decompose::extract_gpu_resnet_segments_with_relu_names(
                self,
                input,
                output_node,
                ibp_bounds,
                ibp_bounds,
                Some(alpha_state),
            )?;
        let od = spec_row.len();
        if od == 0 || od > 4096 {
            return None;
        }
        // Masked pre-activation lower per ReLU in FOLD order (mirrors the warmup-gradient
        // path: zero where the neuron is not unstable, incl. #4404 channel→spatial expand).
        let mut pre_lowers: Vec<Vec<f32>> = Vec::with_capacity(relu_names.len());
        for name in &relu_names {
            let node = self.nodes.get(name)?;
            let input_name = node.inputs.first()?;
            let pre = if input_name == NETWORK_INPUT {
                input
            } else {
                ibp_bounds.get(input_name)?
            };
            let pre_l = pre.lower().as_slice()?;
            let pl = match alpha_state.relu_unstable_mask(name) {
                Some(mask_raw) => {
                    let mask = if alpha_state.spatial_shape(name).is_some() {
                        alpha_state.expand_mask(name, mask_raw)
                    } else {
                        mask_raw.clone()
                    };
                    let m = mask.as_slice()?;
                    if m.len() != pre_l.len() {
                        return None;
                    }
                    pre_l
                        .iter()
                        .zip(m.iter())
                        .map(|(&l, &unstable)| if unstable { l } else { 0.0 })
                        .collect::<Vec<f32>>()
                }
                None => vec![0.0f32; pre_l.len()],
            };
            pre_lowers.push(pl);
        }
        let seed = ny_core::GpuCrownSeed {
            lower_a: spec_row.to_vec().into(),
            upper_a: spec_row.to_vec().into(),
            lower_b: vec![0.0f32].into(),
            upper_b: vec![0.0f32].into(),
            num_specs: 1,
            current_dim: od,
        };
        let in_lo: Vec<f32> = input.lower().iter().copied().collect();
        let in_hi: Vec<f32> = input.upper().iter().copied().collect();
        if deadline.is_some_and(|value| Instant::now() >= value) {
            return None;
        }
        let result = {
            let _deadline_scope =
                crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(gpu, deadline);
            gpu.crown_backward_gpu_resnet_sound_grad(
                &segments,
                &seed,
                &in_lo,
                &in_hi,
                &pre_lowers,
                &frontier_abs,
                &node_abs,
            )
            .ok()?
        };
        if deadline.is_some_and(|value| Instant::now() >= value) {
            return None;
        }
        if result.relu_grads.len() != relu_names.len()
            || result
                .relu_grads
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
            || !crate::sound_gpu_gate::gpu_interval_payload_is_publishable(
                &result.lower_bounds,
                &result.upper_bounds,
                1,
            )
        {
            return None;
        }
        // #root-alpha-true (dark, NY_ROOT_ALPHA_TRUE=1, default-OFF ⇒ byte-identical):
        // swap the wrong local-rule spec-objective gradient for the TRUE per-neuron
        // chain-rule gradient `max(ν_i,0)·ĥ_i(x*)` of THIS margin row's lower bound
        // (single replay: `num_specs=1`, so `result.lower_bounds[0]` is the GPU fold
        // lb the replay must reproduce). This is the margin-targeted twin of the
        // identity-seed warmup fix — it optimizes α directly for the spec row (the
        // objective the root LP oracle is defined over). Fail-closed to the local
        // grads on any miss (gradients are non-soundness-critical).
        let relu_grads_fold: Vec<Vec<f32>> =
            if crate::network::graph_alpha::resnet_decompose::multiobj_joint_alpha_enabled() {
                // #multiobj-joint-alpha (NY_MULTIOBJ_JOINT_ALPHA=1): the TRUE JOINT
                // α-gradient of THIS disjunct's spec-row lower bound via the design-doc
                // reverse-mode adjoint (`joint_alpha_gradient`, num_specs=1 seed=spec_row).
                // The local rule this replaces is stuck ~0.23 above ny's own relaxation LP
                // optimum on the cifar100 straggler. Fail-closed to the local grads on any
                // miss — sound either way, the gradient only steers α ∈ [0,1].
                //
                // #multiobj-joint-alpha-gpu (NY_MULTIOBJ_JOINT_ALPHA_GPU=1): compute that same
                // true-joint gradient on the GPU-resident adjoint first (no CPU whole-network
                // re-fold), so the straggler tightening does not spend the BaB budget.
                // GPU → CPU-oracle → local grads, sound at every rung (steers α∈[0,1]).
                let gpu_joint = if crate::network::graph_alpha::resnet_decompose::multiobj_joint_alpha_gpu_enabled() {
                    if deadline.is_some_and(|value| Instant::now() >= value) {
                        return None;
                    }
                    let result = {
                        let _deadline_scope =
                            crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(gpu, deadline);
                        crate::network::graph_alpha::resnet_decompose::joint_alpha_grads_fold_gpu(
                            gpu,
                            &segments,
                            spec_row,
                            1,
                            od,
                            &in_lo,
                            &in_hi,
                            &pre_lowers,
                            relu_names.len(),
                        )
                    };
                    if deadline.is_some_and(|value| Instant::now() >= value) {
                        return None;
                    }
                    result
                } else {
                    None
                };
                match gpu_joint {
                    Some(g) => g,
                    None => {
                        match crate::network::graph_alpha::resnet_decompose::joint_alpha_grads_fold(
                            &segments,
                            spec_row,
                            &[0.0f32],
                            1,
                            od,
                            &in_lo,
                            &in_hi,
                            &pre_lowers,
                            relu_names.len(),
                        ) {
                            Some(g) => g,
                            None => result.relu_grads,
                        }
                    }
                }
            } else if std::env::var("NY_ROOT_ALPHA_TRUE").ok().as_deref() == Some("1") {
                let gpu_lb = result.lower_bounds.first().copied().unwrap_or(f32::NAN);
                // #true-grad-gpu-replay: run the TRUE-gradient replay's backward
                // walk on the SAME armed sound backend that produced `result`
                // (real certified-error tables), instead of the per-iteration
                // full CPU replay. Fail-closed: any refusal — including the
                // replay-vs-fold lb tolerance oracle — takes the byte-identical
                // CPU replay path inside.
                let gpu_replay_ops =
                    crate::beta_crown::engine::graph::propagation::batched::wide_alpha_true::TrueGradGpuReplayOps::new(
                        gpu, &frontier_abs, &node_abs,
                    );
                // BEHAVIOR NOTE (review defect 2, accepted as an improvement):
                // HEAD called the deadline-free face here, so a CPU replay
                // could complete LATE past the verifier deadline. Threading
                // `deadline` means an expiring budget now aborts the replay
                // mid-walk and falls to the local gradient — the sound,
                // budget-honest direction for scored runs.
                match crate::beta_crown::engine::graph::propagation::batched::wide_alpha_true::true_alpha_grads_for_row_gpu_until(
                gpu_replay_ops.as_ref(),
                &segments, spec_row, &[], &in_lo, &in_hi, relu_names.len(), gpu_lb,
                crate::beta_gpu_probe_armed(),
                deadline,
            ) {
                Some(mut g) => {
                    // Mask stable neurons (pre_lower==0), matching the local path.
                    for (r, gr) in g.iter_mut().enumerate() {
                        if let Some(pl) = pre_lowers.get(r) {
                            if pl.len() == gr.len() {
                                for (gi, &plv) in gr.iter_mut().zip(pl.iter()) {
                                    if plv == 0.0 {
                                        *gi = 0.0;
                                    }
                                }
                            }
                        }
                    }
                    g
                }
                None => result.relu_grads,
            }
            } else {
                result.relu_grads
            };
        let mut relu = std::collections::BTreeMap::new();
        for (name, grad) in relu_names.iter().zip(relu_grads_fold) {
            relu.insert(name.clone(), Array1::from(grad));
        }
        debug!(
            relus = relu_names.len(),
            "Per-disjunct α-CROWN: GPU analytic spec-objective gradients (step 3b-ii)"
        );
        if crate::beta_gpu_probe_armed() {
            eprintln!("[spec-obj-grad] GPU SUCCESS relus={}", relu_names.len());
        }
        Some(GpuSpecObjectiveEvaluation {
            lower_bound: result.lower_bounds.first().copied().unwrap_or(f32::NAN),
            gradients: DagSpsaGradients {
                relu,
                monotone: std::collections::BTreeMap::new(),
                sqrt: std::collections::BTreeMap::new(),
                reciprocal: std::collections::BTreeMap::new(),
            },
        })
    }
}

#[cfg(test)]
mod direct_objective_tests {
    use super::*;
    use crate::bounds::Optimizer;
    use crate::network::graph_alpha::resnet_skeleton::test_support::{
        box_input, conv_resnet_fixture, mk_alpha, CONV_FIXTURE_RELUS,
    };
    use ndarray::Array1;
    use ny_core::{
        GemmEngine, GpuCrownBackward, GpuCrownGradResult, GpuCrownLayer, GpuCrownResult,
        GpuCrownSeed, GpuResnetSegment, NaiveCpuGemmEngine,
    };
    use ny_test_utils::env::with_env_edits;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Duration;

    /// Deterministic stand-in for the sound GPU direct-C fold. Each call pairs
    /// one scripted lower bound with a call-tagged gradient, which lets the
    /// tests prove that the objective and gradient came from one backend call.
    struct ScriptedDirectEngine {
        objectives: Vec<f32>,
        gradient: f32,
        nonfinite_gradient_call: Option<usize>,
        calls: AtomicUsize,
        cooperative_deadline: bool,
        deadline_writes: Mutex<Vec<Option<Instant>>>,
    }

    impl ScriptedDirectEngine {
        fn new(objectives: Vec<f32>, gradient: f32) -> Self {
            Self {
                objectives,
                gradient,
                nonfinite_gradient_call: None,
                calls: AtomicUsize::new(0),
                cooperative_deadline: false,
                deadline_writes: Mutex::new(Vec::new()),
            }
        }

        fn with_deadline_support(mut self) -> Self {
            self.cooperative_deadline = true;
            self
        }

        fn with_nonfinite_gradient_call(mut self, call: usize) -> Self {
            self.nonfinite_gradient_call = Some(call);
            self
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        fn deadline_writes(&self) -> Vec<Option<Instant>> {
            self.deadline_writes
                .lock()
                .expect("deadline_writes mutex")
                .clone()
        }
    }

    impl GemmEngine for ScriptedDirectEngine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }

        fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
            Some(self)
        }
    }

    impl GpuCrownBackward for ScriptedDirectEngine {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp(
                "scripted engine: direct resnet gradient fold only".to_string(),
            ))
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            true
        }

        fn honors_crown_backward_deadline(&self) -> bool {
            self.cooperative_deadline
        }

        fn set_crown_backward_deadline(&self, deadline: Option<Instant>) {
            self.deadline_writes
                .lock()
                .expect("deadline_writes mutex")
                .push(deadline);
        }

        fn crown_backward_gpu_resnet_sound_grad(
            &self,
            _segments: &[GpuResnetSegment],
            seed: &GpuCrownSeed,
            _input_lower: &[f32],
            _input_upper: &[f32],
            relu_pre_lower: &[Vec<f32>],
            _frontier_abs: &[Vec<f32>],
            _node_abs: &[Vec<f32>],
        ) -> Result<GpuCrownGradResult> {
            assert_eq!(seed.num_specs, 1, "direct objective must seed one spec row");
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let objective = self
                .objectives
                .get(call)
                .copied()
                .unwrap_or_else(|| *self.objectives.last().expect("scripted objective"));
            let tagged_gradient = if self.nonfinite_gradient_call == Some(call) {
                f32::NAN
            } else {
                self.gradient + call as f32
            };
            Ok(GpuCrownGradResult {
                lower_bounds: vec![objective],
                upper_bounds: vec![objective + 1.0],
                relu_grads: relu_pre_lower
                    .iter()
                    .map(|pre| vec![tagged_gradient; pre.len()])
                    .collect(),
            })
        }
    }

    fn fixture() -> (
        GraphNetwork,
        BoundedTensor,
        std::collections::HashMap<String, BoundedTensor>,
        GraphAlphaState,
        Vec<f32>,
    ) {
        let graph = conv_resnet_fixture();
        let input = box_input(&[2, 6, 6], -1.0, 1.0);
        let bounds = graph.collect_node_bounds(&input).expect("fixture bounds");
        let alpha = mk_alpha(&graph, &bounds, &CONV_FIXTURE_RELUS, 0.35, 0.65);
        let output_dim = bounds
            .get("conv_out")
            .expect("fixture output")
            .lower()
            .len();
        let mut spec = vec![0.0; output_dim];
        spec[0] = 1.0;
        (graph, input, bounds, alpha, spec)
    }

    fn direct_test_env(f: impl FnOnce()) {
        with_env_edits(|env| {
            env.set("NY_SPEC_ALPHA_DIRECT", "1");
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA_GPU");
            env.remove("NY_ROOT_ALPHA_TRUE");
            f();
        });
    }

    fn one_sgd_step(
        mut state: GraphAlphaState,
        learning_rate: f32,
        gradient: f32,
    ) -> GraphAlphaState {
        let nodes: Vec<(String, usize)> = state
            .alphas
            .iter()
            .map(|(name, alpha)| (name.clone(), alpha.len()))
            .collect();
        for (name, len) in nodes {
            let neg_gradient = Array1::from_elem(len, -gradient);
            state.update(&name, &neg_gradient, learning_rate, 0.0);
            state.update_upper(&name, &neg_gradient, learning_rate, 0.0);
        }
        state
    }

    fn assert_relu_optimizer_state_eq(actual: &GraphAlphaState, expected: &GraphAlphaState) {
        assert_eq!(actual.alphas, expected.alphas, "lower alpha snapshot");
        assert_eq!(
            actual.alphas_upper, expected.alphas_upper,
            "upper alpha snapshot"
        );
        assert_eq!(actual.velocity, expected.velocity, "lower SGD state");
        assert_eq!(
            actual.velocity_upper, expected.velocity_upper,
            "upper SGD state"
        );
    }

    #[test]
    fn spec_alpha_direct_gate_is_exact_default_off() {
        assert!(!spec_alpha_direct_enabled_for(None), "unset must be OFF");
        assert!(!spec_alpha_direct_enabled_for(Some("0")), "0 must be OFF");
        assert!(!spec_alpha_direct_enabled_for(Some("true")));
        assert!(!spec_alpha_direct_enabled_for(Some("")));
        assert!(spec_alpha_direct_enabled_for(Some("1")), "1 must be ON");
    }

    #[test]
    fn direct_fold_returns_paired_objective_and_gradient_from_one_call() {
        direct_test_env(|| {
            let (graph, input, bounds, alpha, spec) = fixture();
            let engine = ScriptedDirectEngine::new(vec![7.25], 3.5);
            let evaluation = graph
                .try_gpu_spec_objective_evaluation(
                    &input,
                    &bounds,
                    &alpha,
                    "conv_out",
                    &spec,
                    Some(&engine),
                    None,
                )
                .expect("fixture must use direct GPU fold");

            assert_eq!(engine.calls(), 1, "objective+gradient need one fold");
            assert_eq!(evaluation.lower_bound.to_bits(), 7.25f32.to_bits());
            assert_eq!(evaluation.gradients.relu.len(), CONV_FIXTURE_RELUS.len());
            for gradient in evaluation.gradients.relu.values() {
                assert!(gradient.iter().all(|&value| value == 3.5));
            }
        });
    }

    #[test]
    fn direct_fold_obeys_deadline_admission_and_exact_scope() {
        direct_test_env(|| {
            let (graph, input, bounds, alpha, spec) = fixture();

            let expired_engine = ScriptedDirectEngine::new(vec![7.25], 3.5).with_deadline_support();
            let expired = graph.try_gpu_spec_objective_evaluation(
                &input,
                &bounds,
                &alpha,
                "conv_out",
                &spec,
                Some(&expired_engine),
                Some(
                    Instant::now()
                        .checked_sub(Duration::from_millis(1))
                        .expect("one millisecond fits before the current instant"),
                ),
            );
            assert!(expired.is_none());
            assert_eq!(expired_engine.calls(), 0);
            assert!(expired_engine.deadline_writes().is_empty());

            let noncoop_engine = ScriptedDirectEngine::new(vec![7.25], 3.5);
            let noncoop = graph.try_gpu_spec_objective_evaluation(
                &input,
                &bounds,
                &alpha,
                "conv_out",
                &spec,
                Some(&noncoop_engine),
                Some(Instant::now() + Duration::from_secs(30)),
            );
            assert!(noncoop.is_none());
            assert_eq!(noncoop_engine.calls(), 0);

            let deadline = Instant::now() + Duration::from_secs(30);
            let live_engine = ScriptedDirectEngine::new(vec![7.25], 3.5).with_deadline_support();
            let live = graph.try_gpu_spec_objective_evaluation(
                &input,
                &bounds,
                &alpha,
                "conv_out",
                &spec,
                Some(&live_engine),
                Some(deadline),
            );
            assert!(live.is_some());
            assert_eq!(live_engine.calls(), 1);
            assert_eq!(live_engine.deadline_writes(), vec![Some(deadline), None]);
        });
    }

    #[test]
    fn direct_optimizer_returns_alpha_at_best_evaluated_bound() {
        direct_test_env(|| {
            let (graph, input, bounds, initial, spec) = fixture();
            // Evaluated states: initial -> one update -> two updates. The
            // middle state has the best direct bound and must be returned.
            let engine = ScriptedDirectEngine::new(vec![1.0, 3.0, 2.0], 1.0);
            let expected = one_sgd_step(initial.clone(), 0.1, 1.0);
            let config = AlphaCrownConfig {
                iterations: 3,
                optimizer: Optimizer::Sgd,
                learning_rate: 0.1,
                lr_decay: 1.0,
                momentum: 0.0,
                ..AlphaCrownConfig::default()
            };

            let actual = graph
                .optimize_alpha_for_spec_objective(
                    &input,
                    &bounds,
                    &initial,
                    &config,
                    &spec,
                    Some(&engine),
                )
                .expect("direct optimization");

            assert_eq!(engine.calls(), 3, "one direct fold per iteration");
            assert_relu_optimizer_state_eq(&actual, &expected);
        });
    }

    #[test]
    fn nonfinite_direct_bound_discards_snapshot_and_falls_back() {
        direct_test_env(|| {
            let (graph, input, bounds, initial, spec) = fixture();
            // Iteration 0 succeeds and updates once. Iteration 1's +Inf must
            // permanently select the legacy path and return the current state,
            // not the stale direct-best snapshot taken before the update.
            let engine = ScriptedDirectEngine::new(vec![1.0, f32::INFINITY], 1.0);
            let expected = one_sgd_step(initial.clone(), 0.1, 1.0);
            let config = AlphaCrownConfig {
                iterations: 2,
                optimizer: Optimizer::Sgd,
                learning_rate: 0.1,
                lr_decay: 1.0,
                momentum: 0.0,
                ..AlphaCrownConfig::default()
            };

            let actual = graph
                .optimize_alpha_for_spec_objective(
                    &input,
                    &bounds,
                    &initial,
                    &config,
                    &spec,
                    Some(&engine),
                )
                .expect("legacy fallback after nonfinite direct bound");

            assert_eq!(engine.calls(), 2);
            assert_relu_optimizer_state_eq(&actual, &expected);
        });
    }

    #[test]
    fn nonfinite_direct_gradient_resumes_legacy_update_path() {
        direct_test_env(|| {
            let (graph, input, bounds, initial, spec) = fixture();
            // Call 1 has a finite direct bound but a NaN gradient. Direct mode
            // must reject the entire paired evaluation, then the legacy
            // gradient site retries (call 2) and advances the current state.
            let engine =
                ScriptedDirectEngine::new(vec![1.0, 2.0, 3.0], 1.0).with_nonfinite_gradient_call(1);
            let expected = one_sgd_step(one_sgd_step(initial.clone(), 0.1, 1.0), 0.1, 3.0);
            let config = AlphaCrownConfig {
                iterations: 3,
                optimizer: Optimizer::Sgd,
                learning_rate: 0.1,
                lr_decay: 1.0,
                momentum: 0.0,
                ..AlphaCrownConfig::default()
            };

            let actual = graph
                .optimize_alpha_for_spec_objective(
                    &input,
                    &bounds,
                    &initial,
                    &config,
                    &spec,
                    Some(&engine),
                )
                .expect("legacy update after nonfinite direct gradient");

            assert_eq!(engine.calls(), 3);
            assert_relu_optimizer_state_eq(&actual, &expected);
        });
    }
}

#[cfg(test)]
mod intersect_ibp_tests {
    //! Unit tests for the per-iteration "intersect the raw backward output bound
    //! with the always-available IBP output bound" tightening applied at the three
    //! alpha-optimization sites (alpha.rs, warm_start.rs, alpha_spec_objective.rs).
    //!
    //! These exercise the exact `intersection_per_element`-based shadowing pattern
    //! inserted at each site, decoupled from the full network plumbing.
    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    /// Mirror the site pattern: given a raw CROWN `output_bounds` and an IBP map
    /// keyed by `output_node`, return the (possibly tightened) bound the way each
    /// site computes it.
    fn intersect_with_ibp(
        output_bounds: BoundedTensor,
        ibp: &std::collections::HashMap<String, BoundedTensor>,
        output_node: &str,
    ) -> BoundedTensor {
        match ibp.get(output_node) {
            Some(ibp_out) if ibp_out.shape() == output_bounds.shape() => output_bounds
                .intersection_per_element(ibp_out)
                .map(|(t, _)| t)
                .unwrap_or(output_bounds),
            _ => output_bounds,
        }
    }

    fn bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn()).unwrap()
    }

    #[test]
    fn loose_crown_capped_by_tighter_ibp_elementwise() {
        // Exploded (loose) CROWN output bound, finite (tighter-on-every-element) IBP.
        let crown = bt(&[-100.0, -50.0, -30.0], &[100.0, 50.0, 30.0]);
        let ibp = bt(&[-5.0, -10.0, -8.0], &[5.0, 10.0, 8.0]);

        let mut map = std::collections::HashMap::new();
        map.insert("out".to_string(), ibp.clone());

        let result = intersect_with_ibp(crown.clone(), &map, "out");

        // Per-element: lower = max(crown_l, ibp_l), upper = min(crown_u, ibp_u).
        for i in 0..3 {
            let exp_l = crown.lower()[i].max(ibp.lower()[i]);
            let exp_u = crown.upper()[i].min(ibp.upper()[i]);
            assert_eq!(result.lower()[i], exp_l, "lower[{i}] tightened");
            assert_eq!(result.upper()[i], exp_u, "upper[{i}] tightened");
            // SOUND: result never looser than either operand on overlap.
            assert!(result.lower()[i] >= crown.lower()[i]);
            assert!(result.upper()[i] <= crown.upper()[i]);
            assert!(result.lower()[i] >= ibp.lower()[i]);
            assert!(result.upper()[i] <= ibp.upper()[i]);
            // Valid interval (l <= u).
            assert!(result.lower()[i] <= result.upper()[i]);
        }
        // Concretely on this data the IBP wins every element.
        assert_eq!(result.lower(), ibp.lower());
        assert_eq!(result.upper(), ibp.upper());
    }

    #[test]
    fn already_tighter_crown_is_unchanged() {
        // CROWN is already tighter than IBP on every element → no-op.
        let crown = bt(&[-1.0, -2.0], &[1.0, 2.0]);
        let ibp = bt(&[-5.0, -5.0], &[5.0, 5.0]);

        let mut map = std::collections::HashMap::new();
        map.insert("out".to_string(), ibp);

        let result = intersect_with_ibp(crown.clone(), &map, "out");

        // Byte-identical to the original CROWN bound (the intersection picked CROWN
        // on every element).
        assert_eq!(result.lower(), crown.lower());
        assert_eq!(result.upper(), crown.upper());
    }

    #[test]
    fn missing_ibp_key_keeps_crown() {
        let crown = bt(&[-7.0], &[7.0]);
        let map: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        let result = intersect_with_ibp(crown.clone(), &map, "out");
        assert_eq!(result.lower(), crown.lower());
        assert_eq!(result.upper(), crown.upper());
    }

    #[test]
    fn shape_mismatch_keeps_crown() {
        let crown = bt(&[-7.0, 1.0], &[7.0, 2.0]);
        let ibp = bt(&[-1.0], &[1.0]); // different shape

        let mut map = std::collections::HashMap::new();
        map.insert("out".to_string(), ibp);

        let result = intersect_with_ibp(crown.clone(), &map, "out");
        assert_eq!(result.lower(), crown.lower());
        assert_eq!(result.upper(), crown.upper());
    }
}
