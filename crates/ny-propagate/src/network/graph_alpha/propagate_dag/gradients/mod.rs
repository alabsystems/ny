// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Gradient computation dispatch for DAG α-CROWN.
//!
//! Handles SPSA, Finite Differences, Analytic, and AnalyticChain gradient
//! methods for the optimization loop. Each method computes per-ReLU gradients;
//! SPSA additionally accumulates bilinear and mul_binary gradients.

mod finite_differences;
mod spsa;

use crate::bounds::alpha_reciprocal::ReciprocalGradients;
use crate::bounds::{GradientMethod, GraphAlphaState, MonotoneSShapedGradients, SqrtGradients};

use ndarray::{Array1, Array2, Array4};
use ny_core::{GpuCrownSeed, GpuResnetSegment, Result};
use ny_tensor::BoundedTensor;
use std::collections::{BTreeMap, HashMap};
use tracing::debug;

use super::super::runtime_state::DagAlphaRuntimeState;
use super::DagAlphaLoopContext;
use crate::network::core::{GraphNetwork, NETWORK_INPUT};
use crate::network::graph_alpha::resnet_decompose::{
    extract_gpu_resnet_segments_with_relu_names, joint_alpha_grads_fold,
    multiobj_joint_alpha_enabled, multiobj_joint_alpha_gpu_enabled, resnet_warmup_gpu_enabled,
    root_alpha_gpu_enabled,
};

/// Result of the gradient dispatch computation.
///
/// Contains computed gradients for all node types. The caller uses these
/// to drive the supplement computation and alpha update steps.
pub(super) struct GradientDispatchResult {
    pub(super) numerical_gradients: Vec<Array1<f32>>,
    /// Upper-path gradients. `None` means same as `numerical_gradients` (#2297:
    /// avoids a full `Vec<Array1<f32>>` clone for non-Analytic gradient methods).
    pub(super) numerical_gradients_upper: Option<Vec<Array1<f32>>>,
    pub(super) bilinear_grads: HashMap<String, Array4<f32>>,
    pub(super) mul_binary_grads: HashMap<String, Array2<f32>>,
    pub(super) s_shaped_grads: BTreeMap<String, MonotoneSShapedGradients>,
    pub(super) sqrt_grads: BTreeMap<String, SqrtGradients>,
    pub(super) reciprocal_grads: BTreeMap<String, ReciprocalGradients>,
}

/// #root-alpha-gpu (B): one iteration's loop-top GPU warmup fold, kept so the
/// gradient site can consume the SAME kernel call's outputs instead of running
/// a second extraction + fold in the same iteration.
///
/// Alpha-currency argument for the reuse: α only changes at the END of an
/// iteration (`alpha_update::update_all_alphas`, called after the gradient
/// site in `dag_alpha_optimize_loop`), so the α folded into the loop-top
/// segments is still current when the gradient site runs. `refresh_fired`
/// (set by the reference-bound refresh block) and a stale `iter` both force a
/// fresh gradient-site fold instead — fail closed. Reuse only steers α (any
/// α ∈ [0,1] is a sound relaxation); it can never decide a verdict.
pub(super) struct WarmupGpuIterCache {
    /// The optimization-loop iteration this fold belongs to (stamped by the
    /// loop that owns the cache).
    pub(super) iter: usize,
    /// The fold's segments (skeleton-folded under increment A, so the static
    /// weight payloads are shared `Arc`s).
    pub(super) segments: Vec<GpuResnetSegment>,
    /// Fold-order per-ReLU node names.
    pub(super) relu_names: Vec<String>,
    /// Masked pre-activation lower bounds per ReLU (fold order).
    pub(super) pre_lowers: Vec<Vec<f32>>,
    /// Per-ReLU local-rule α gradients captured by the kernel (fold order).
    pub(super) relu_grads: Vec<Vec<f32>>,
    /// Per-output-row lower bounds (the TRUE replay's fail-closed validation
    /// input).
    pub(super) lower_bounds: Vec<f32>,
    /// Set when the reference-bound refresh fired this iteration —
    /// invalidates gradient reuse for the rest of the iteration.
    pub(super) refresh_fired: bool,
}

/// Prepare optional references to bilinear and mul_binary alpha maps.
///
/// Eliminates the repeated `if has_bilinear { Some(&alphas) } else { None }` pattern
/// that appears across gradient and supplement computations.
#[allow(clippy::type_complexity)]
pub(super) fn alpha_refs<'a>(
    ctx: &DagAlphaLoopContext<'_>,
    bilinear_alphas: &'a HashMap<String, Array4<f32>>,
    mul_binary_alphas: &'a HashMap<String, Array2<f32>>,
) -> (
    Option<&'a HashMap<String, Array4<f32>>>,
    Option<&'a HashMap<String, Array2<f32>>>,
) {
    let bl = if ctx.has_bilinear {
        Some(bilinear_alphas)
    } else {
        None
    };
    let mb = if ctx.has_mul_binary {
        Some(mul_binary_alphas)
    } else {
        None
    };
    (bl, mb)
}

/// Create zero-valued scratch gradient buffers matching the shape of existing gradients.
/// #root-alpha-true gate (dark, default OFF ⇒ byte-identical): use the TRUE
/// per-neuron chain-rule warmup α gradient instead of the wrong local rule.
fn root_alpha_true_enabled() -> bool {
    std::env::var("NY_ROOT_ALPHA_TRUE").ok().as_deref() == Some("1")
}

fn zeros_like_gradients(gradients: &[Array1<f32>]) -> Vec<Array1<f32>> {
    gradients.iter().map(|g| Array1::zeros(g.len())).collect()
}

impl GraphNetwork {
    /// Compute gradients for all node types using the configured gradient method.
    ///
    /// Dispatches to SPSA, Finite Differences, Analytic, or AnalyticChain based on
    /// `config.gradient_method`. Initializes gradient accumulators and returns them
    /// in `GradientDispatchResult` for use by supplements and alpha updates.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_dag_gradients(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &mut DagAlphaRuntimeState,
        bilinear_alphas: &mut HashMap<String, Array4<f32>>,
        mul_binary_alphas: &mut HashMap<String, Array2<f32>>,
        gradients: &[Array1<f32>],
        gradients_upper: &[Array1<f32>],
        eps: f32,
        iter: usize,
        warmup_cache: Option<&WarmupGpuIterCache>,
    ) -> Result<GradientDispatchResult> {
        // Initialize gradient accumulators.
        // Bilinear gradient accumulator — populated by SPSA, left as zeros for other
        // methods. Available for the bilinear Adam update below (#3287).
        let mut bilinear_grads: HashMap<String, Array4<f32>> = bilinear_alphas
            .iter()
            .map(|(name, alpha)| (name.clone(), Array4::zeros(alpha.raw_dim())))
            .collect();
        // MulBinary gradient accumulator (#3439 Phase 3).
        let mut mul_binary_grads: HashMap<String, Array2<f32>> = mul_binary_alphas
            .iter()
            .map(|(name, alpha)| (name.clone(), Array2::zeros(alpha.raw_dim())))
            .collect();
        let s_shaped_grads: BTreeMap<String, MonotoneSShapedGradients> = runtime
            .graph()
            .monotone_alpha_names()
            .map(|name| {
                let alpha = runtime
                    .graph()
                    .monotone_s_shaped_alpha(name)
                    .ok_or_else(|| {
                        ny_core::NyError::InternalError(
                            "monotone alpha name iterator must point to existing state".into(),
                        )
                    })?;
                Ok((name.clone(), alpha.zeros_gradients()))
            })
            .collect::<Result<_>>()?;
        let sqrt_grads: BTreeMap<String, SqrtGradients> = runtime
            .graph()
            .sqrt_alpha_names()
            .map(|name| {
                let alpha = runtime.graph().sqrt_alpha(name).ok_or_else(|| {
                    ny_core::NyError::InternalError(
                        "sqrt alpha name iterator must point to existing state".into(),
                    )
                })?;
                Ok((name.clone(), alpha.zeros_gradients()))
            })
            .collect::<Result<_>>()?;
        let reciprocal_grads: BTreeMap<String, ReciprocalGradients> = runtime
            .graph()
            .reciprocal_alpha_names()
            .map(|name| {
                let alpha = runtime.graph().reciprocal_alpha(name).ok_or_else(|| {
                    ny_core::NyError::InternalError(
                        "reciprocal alpha name iterator must point to existing state".into(),
                    )
                })?;
                Ok((name.clone(), alpha.zeros_gradients()))
            })
            .collect::<Result<_>>()?;

        let num_relus = ctx.relu_nodes.len();
        // GPU-resident resnet warmup fast-path (#unsat-keystone step 3b-ii): for
        // AnalyticChain on a decomposable ResNet suffix, get the per-ReLU gradients
        // from ONE GPU resident backward instead of the slow CPU intermediates pass.
        // `None` (no sound GPU engine / non-resnet / any mismatch) → CPU path below.
        let gpu_warmup_gradients: Option<Vec<Array1<f32>>> =
            if matches!(ctx.config.gradient_method, GradientMethod::AnalyticChain) {
                self.try_gpu_resnet_warmup_gradients(
                    ctx,
                    node_bounds,
                    runtime,
                    gradients,
                    warmup_cache,
                    iter,
                )
            } else {
                None
            };
        // Compute gradient using configured method.
        //
        // NOTE: DAG α-CROWN historically ignored `config.gradient_method` and always used the
        // per-ReLU local gradients returned by `propagate_linear_with_alpha`. Honor the config
        // so the default (`AnalyticChain`) is actually used on ResNet-like graphs with skip
        // connections.
        let numerical_gradients: Vec<Array1<f32>> = match gpu_warmup_gradients {
            Some(gpu_grads) => gpu_grads,
            None => match ctx.config.gradient_method {
                GradientMethod::Spsa => self.compute_spsa_gradients(
                    ctx,
                    node_bounds,
                    runtime,
                    bilinear_alphas,
                    mul_binary_alphas,
                    &mut bilinear_grads,
                    &mut mul_binary_grads,
                    gradients,
                    num_relus,
                    eps,
                )?,
                GradientMethod::FiniteDifferences => self.compute_fd_gradients(
                    ctx,
                    node_bounds,
                    runtime,
                    bilinear_alphas,
                    mul_binary_alphas,
                    gradients,
                    num_relus,
                    eps,
                )?,
                GradientMethod::Analytic => {
                    // Local gradients from CROWN backward pass
                    gradients.to_vec()
                }
                GradientMethod::AnalyticChain => {
                    // True chain-rule gradients using intermediate A matrices.
                    // Run backward pass that stores A matrices at each ReLU node.
                    let mut scratch = zeros_like_gradients(gradients);
                    let mut scratch_upper = zeros_like_gradients(gradients);
                    // Pass current bilinear/mul_binary alphas through backward call (#3287, #3439).
                    let (bl, mb) = alpha_refs(ctx, bilinear_alphas, mul_binary_alphas);

                    match self.dag_alpha_backward_pass_with_intermediates(
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
                        bl,
                        mb,
                        // #cgan-alpha-amatrix: NO deadline for the AnalyticChain
                        // gradient backward — mirror the sequential twin
                        // (sequential_gradients.rs, #3393). Handing it the global
                        // deadline made the FIRST deadline-aware node in reverse
                        // order (e.g. the output Gemm) abort once the budget was
                        // spent by the collector/refresh, dropping the A-matrices for
                        // EVERY downstream ReLU (all 7 cgan generator ReLUs) → zero-
                        // length chain gradients → frozen alpha. A truncated gradient
                        // records no A-matrices and is useless anyway, so an
                        // unbounded-but-complete pass strictly dominates; the
                        // output_dim-row pass is bounded and the top-level loop
                        // deadline (config.past_deadline) still bounds total time.
                        None,
                    ) {
                        Ok((_bounds, intermediate)) => {
                            // Chain-rule gradients now include Patches-mode ReLUs
                            // because backward.rs stores Dense intermediates for them
                            // (#3293 Approach B). No local gradient fallback needed.
                            let relu_names: Vec<String> = ctx
                                .relu_nodes
                                .iter()
                                .map(|(name, _)| name.clone())
                                .collect();
                            let chain_grads = self.compute_graph_chain_rule_gradients(
                                ctx.input,
                                &relu_names,
                                &intermediate,
                            );
                            // #4404: reduce per-neuron chain-rule gradients to per-channel
                            // for nodes using channel-only alpha (full_conv_alpha: False).
                            //
                            // #cgan-alpha-amatrix robustness: if a ReLU's A-matrix was
                            // still missing (partial map -> empty chain gradient),
                            // substitute the local per-ReLU gradient `gradients[i]` —
                            // already in final reduced form, exactly what the Err
                            // branch below uses via `gradients.to_vec()` — so alpha
                            // keeps moving via the local gradient instead of freezing
                            // on a zero-length gradient (the length-mismatch skip in
                            // GraphAlphaState::update).
                            chain_grads
                                .into_iter()
                                .zip(relu_names.iter())
                                .enumerate()
                                .map(|(i, (grad, name))| {
                                    if grad.is_empty() {
                                        gradients[i].clone()
                                    } else {
                                        runtime.graph().reduce_gradient(name, &grad)
                                    }
                                })
                                .collect()
                        }
                        Err(e) => {
                            // Fall back to local gradients if intermediate storage failed
                            if iter == 0 {
                                debug!(
                                    "DAG α-CROWN: AnalyticChain failed ({}), using local gradients",
                                    e
                                );
                            }
                            gradients.to_vec()
                        }
                    }
                }
            },
        };

        // Upper-path gradients (#3393). Analytic method has real upper gradients
        // from the backward pass. Other methods use the same gradient for both paths
        // (SPSA/FiniteDiff perturb jointly, AnalyticChain doesn't separate paths).
        // Non-Analytic methods store `None` and fall back to `numerical_gradients`
        // at the call site, avoiding a full Vec<Array1<f32>> clone (#2297).
        let numerical_gradients_upper: Option<Vec<Array1<f32>>> = match ctx.config.gradient_method {
            GradientMethod::Analytic => Some(gradients_upper.to_vec()),
            _ => None,
        };

        Ok(GradientDispatchResult {
            numerical_gradients,
            numerical_gradients_upper,
            bilinear_grads,
            mul_binary_grads,
            s_shaped_grads,
            sqrt_grads,
            reciprocal_grads,
        })
    }

    /// GPU-resident warmup gradients for ResNet suffixes (cifar100/tinyimagenet unsat
    /// keystone, step 3b-ii). When the output suffix decomposes into GPU resnet
    /// segments, compute the per-ReLU analytic alpha gradients on the GPU in one
    /// resident backward instead of the slow CPU `dag_alpha_backward_pass_with_intermediates`
    /// (which makes the warmup overrun → 0 BaB domains at ≤400s).
    ///
    /// Returns `None` on ANY mismatch or absence (no sound GPU engine, non-decomposable
    /// suffix, oversized seed, size mismatch, GPU error, NaN) so the caller falls back
    /// to the proven CPU path. Gradients are non-soundness-critical (they only steer
    /// alpha — any alpha is a sound relaxation), so even a wrong mapping yields a sound,
    /// if looser, bound; the soundness gate is never at risk here.
    #[allow(clippy::too_many_arguments)]
    fn try_gpu_resnet_warmup_gradients(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &DagAlphaRuntimeState,
        local_gradients: &[Array1<f32>],
        cache: Option<&WarmupGpuIterCache>,
        iter: usize,
    ) -> Option<Vec<Array1<f32>>> {
        // Default ON, opt out with NY_RESNET_WARMUP_GPU=0 (#unsat-keystone step 3b-ii).
        // The two open items from the WIP phase were both settled by measurement:
        // (1) SEED semantics (identity over output_dim) verified end-to-end — R-beta-7d
        // confirmed this hook FIRES on cifar100 (ENTER ×7, gpu=true, extract=true) and
        // the on-device gradient matches the CPU analytic formula
        // (crown_alpha_gradient_resident_matches_cpu_formula); (2) the disjunctive
        // multi-objective warmup got its own hook (try_gpu_spec_objective_gradients,
        // R-beta-7c). Combined with the GPU warmup BOUND (R-beta-8) this broke the
        // cifar100 warmup wall: BaB 0 → >0 domains explored. Non-soundness-critical:
        // gradients only steer alpha (any alpha ∈ [0,1] is a sound relaxation).
        if !resnet_warmup_gpu_enabled() {
            return None;
        }
        let probe = std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1");
        if probe {
            eprintln!(
                "[warmup-grad] ENTER gpu={} od={}",
                ctx.engine
                    .and_then(|e| e.as_gpu_crown_backward())
                    .map(|g| g.provides_sound_gpu_crown())
                    .unwrap_or(false),
                ctx.output_dim
            );
        }
        let gpu = ctx
            .engine
            .and_then(|e| e.as_gpu_crown_backward())
            .filter(|g| g.provides_sound_gpu_crown())?;

        // #root-alpha-gpu (B): reuse the loop-top full fold's outputs instead
        // of a second extraction + kernel run in the same iteration. Valid
        // because α updates only at the END of the iteration
        // (`alpha_update::update_all_alphas`, propagate_dag/mod.rs), so the
        // α folded into the loop-top segments is still current here. A stale
        // `iter`, a fired reference-bound refresh, or a malformed grads list
        // all reject the cache → fresh fold below (fail closed).
        let cache_hit = if root_alpha_gpu_enabled() {
            cache.filter(|c| {
                c.iter == iter && !c.refresh_fired && c.relu_grads.len() == c.relu_names.len()
            })
        } else {
            None
        };
        if probe {
            eprintln!(
                "[warmup-grad] cache_hit={} iter={iter}",
                cache_hit.is_some()
            );
        }

        // Identity seed over the output dim (the warmup optimizes the output bounds),
        // matching the CPU backward's seeding. Guard objective/seed size.
        let od = ctx.output_dim;
        let seed = identity_warmup_seed(od)?;
        let input = ctx.input;
        let in_lo: Vec<f32> = input.lower().iter().copied().collect();
        let in_hi: Vec<f32> = input.upper().iter().copied().collect();

        // Owned outputs of a fresh fold (cache miss, deferred init); the
        // borrow tuple below is the single downstream view over either source.
        let fresh: (
            Vec<GpuResnetSegment>,
            Vec<String>,
            Vec<Vec<f32>>,
            Vec<Vec<f32>>,
            Vec<f32>,
        );
        let (segments, relu_names, pre_lowers, local_rule_grads, lower_bounds): (
            &[GpuResnetSegment],
            &[String],
            &[Vec<f32>],
            &[Vec<f32>],
            &[f32],
        ) = match cache_hit {
            Some(c) => (
                c.segments.as_slice(),
                c.relu_names.as_slice(),
                c.pre_lowers.as_slice(),
                c.relu_grads.as_slice(),
                c.lower_bounds.as_slice(),
            ),
            None => {
                let output_node = if self.output_node.is_empty() {
                    ctx.exec_order.last()?.clone()
                } else {
                    self.output_node.clone()
                };
                let extracted = self.warmup_segments(ctx, node_bounds, runtime, &output_node);
                if probe {
                    eprintln!(
                        "[warmup-grad] extract={} output_node={}",
                        extracted.is_some(),
                        output_node
                    );
                }
                let (segments, relu_names, frontier_abs, node_abs) = extracted?;
                let pre_lowers =
                    self.warmup_masked_pre_lowers(ctx, node_bounds, runtime.graph(), &relu_names)?;
                let result = gpu
                    .crown_backward_gpu_resnet_sound_grad(
                        &segments,
                        &seed,
                        &in_lo,
                        &in_hi,
                        &pre_lowers,
                        &frontier_abs,
                        &node_abs,
                    )
                    .ok()?;
                if result.relu_grads.len() != relu_names.len()
                    || result
                        .lower_bounds
                        .iter()
                        .chain(result.upper_bounds.iter())
                        .any(|v| v.is_nan())
                {
                    return None;
                }
                fresh = (
                    segments,
                    relu_names,
                    pre_lowers,
                    result.relu_grads,
                    result.lower_bounds,
                );
                (
                    fresh.0.as_slice(),
                    fresh.1.as_slice(),
                    fresh.2.as_slice(),
                    fresh.3.as_slice(),
                    fresh.4.as_slice(),
                )
            }
        };

        // #root-alpha-true (dark, NY_ROOT_ALPHA_TRUE=1, default-OFF ⇒ byte-identical):
        // swap the WRONG local-rule warmup gradient (`pre_lower·Σ_j max(A[j,i],0)`,
        // computed on-GPU by `crown_backward_gpu_resnet_sound_grad`) for the TRUE
        // per-neuron chain-rule gradient `max(ν_i,0)·ĥ_i(x*)`
        // (`wide_alpha_true::true_alpha_grads_for_row`). The identity-seed warmup
        // objective is Σ_r lower(output_r); its true gradient w.r.t. α_i is the SUM
        // over output rows r of that row's per-neuron true gradient (each row has its
        // OWN concretization argmin x*_r), so we replay one row per output. The GPU
        // result's `lower_bounds[r]` is the per-row lower bound the replay must
        // reproduce (fail-closed validation inside `true_alpha_grads_for_row`).
        // Fail-closed to the local-rule grads on any miss (gradients are
        // non-soundness-critical — any α∈[0,1] is a sound relaxation).
        let relu_grads_fold: Vec<Vec<f32>> = if multiobj_joint_alpha_enabled() {
            // #multiobj-joint-alpha (NY_MULTIOBJ_JOINT_ALPHA=1): the TRUE JOINT
            // α-gradient via the design-doc reverse-mode adjoint. The warmup's
            // identity seed (num_specs = od) makes ONE adjoint pass compute the
            // gradient of `Σ_r lower(output_r)` over all output rows at once — the
            // same summed objective the local rule (and true_root_warmup_gradients)
            // target, but through the whole fold (not the stuck single-layer rule).
            // Fail-closed to the local grads on any miss (gradients only steer α).
            //
            // #multiobj-joint-alpha-gpu (NY_MULTIOBJ_JOINT_ALPHA_GPU=1): run that same
            // summed-objective joint adjoint on the GPU-resident kernel first (no CPU
            // whole-network re-fold), so the root warmup tightening (measured
            // −1.31 → −0.689 on the cifar100 straggler) does not spend the BaB budget.
            // GPU → CPU-oracle → local grads, sound at every rung (steers α∈[0,1]).
            let gpu_joint = if multiobj_joint_alpha_gpu_enabled() {
                crate::network::graph_alpha::resnet_decompose::joint_alpha_grads_fold_gpu(
                    gpu,
                    segments,
                    &seed.lower_a,
                    od,
                    od,
                    &in_lo,
                    &in_hi,
                    pre_lowers,
                    relu_names.len(),
                )
            } else {
                None
            };
            match gpu_joint {
                Some(g) => g,
                None => match joint_alpha_grads_fold(
                    segments,
                    &seed.lower_a,
                    &seed.lower_b,
                    od,
                    od,
                    &in_lo,
                    &in_hi,
                    pre_lowers,
                    relu_names.len(),
                ) {
                    Some(g) => g,
                    None => {
                        if probe {
                            eprintln!(
                                "[warmup-grad] joint α-gradient unavailable → local rule (fail-closed)"
                            );
                        }
                        local_rule_grads.to_vec()
                    }
                },
            }
        } else if root_alpha_true_enabled() {
            // #root-alpha-gpu (B): on a cache hit the replays run against the
            // cached `segments`/`lower_bounds` — the same fold the loop-top
            // bound came from, still α-current (see `WarmupGpuIterCache`).
            match self.true_root_warmup_gradients(
                segments,
                relu_names,
                pre_lowers,
                &in_lo,
                &in_hi,
                od,
                lower_bounds,
                probe,
            ) {
                Some(g) => g,
                None => {
                    if probe {
                        eprintln!(
                            "[warmup-grad] TRUE gradient unavailable → local rule (fail-closed)"
                        );
                    }
                    local_rule_grads.to_vec()
                }
            }
        } else {
            local_rule_grads.to_vec()
        };

        // Map fold-order grads → ctx.relu_nodes order; start from the local
        // gradients so any ReLU outside the decomposed suffix keeps a value.
        //
        // Store the RAW per-neuron gradient (#w4-gpu-dag-backward): the alpha
        // update reduces per-neuron → per-channel itself via the idempotent
        // `reduce_gradient` (alpha_update.rs), exactly like the CPU-filled
        // gradients and the spec-objective twin (alpha_spec_objective.rs).
        // Reducing HERE and length-checking against the per-NEURON `out[i]`
        // buffer made the check fail on every channel-only-alpha conv node
        // (full_conv_alpha: false — the cifar100 preset), silently discarding
        // the GPU gradients and falling back to the ~27s/iter CPU
        // AnalyticChain backward.
        let pos: HashMap<&str, usize> = ctx
            .relu_nodes
            .iter()
            .enumerate()
            .map(|(i, (n, _))| (n.as_str(), i))
            .collect();
        let mut out = local_gradients.to_vec();
        if probe {
            // #w4-root-alpha diagnosis: are the GPU relu gradients ever
            // NONZERO? (n_interior=0 after a full warmup means alpha never
            // moved — zero grads and dead updates are indistinguishable
            // without this.)
            let (nz, mx) = relu_grads_fold
                .iter()
                .flatten()
                .fold((0usize, 0.0f32), |(nz, mx), &g| {
                    (nz + usize::from(g != 0.0), mx.max(g.abs()))
                });
            eprintln!("[warmup-grad] relu_grads nz={nz} max_abs={mx:e}");
        }
        for (name, grad) in relu_names.iter().zip(relu_grads_fold) {
            let i = *pos.get(name.as_str())?;
            let grad = Array1::from(grad);
            if grad.len() != out[i].len() {
                return None;
            }
            out[i] = grad;
        }
        debug!(
            relus = relu_names.len(),
            "DAG α-CROWN: GPU-resident resnet warmup gradients (step 3b-ii)"
        );
        Some(out)
    }

    /// TRUE per-neuron warmup α gradient (#root-alpha-true, `NY_ROOT_ALPHA_TRUE=1`).
    ///
    /// The identity-seed warmup objective is `Σ_r lower(output_r)`. Its exact
    /// gradient w.r.t. α_i is `Σ_r ∂lower(output_r)/∂α_i`, and each per-row
    /// derivative is `max(ν_i^r, 0)·ĥ_i(x*_r)` — the closed form the finite-
    /// difference oracle settled (`true_grad_oracle_tests.rs`), NOT the wrong
    /// local rule `pre_lower_i·Σ_j max(A[j,i],0)` the GPU warmup computes. Each
    /// output row `r` has its own concretization argmin `x*_r`, so we replay one
    /// spec row (`e_r`) per output through the domain's own segments and SUM the
    /// per-neuron gradients (fold order = `relu_names` order).
    ///
    /// `gpu_lbs[r]` is the GPU fold's lower bound for output `r`; the host replay
    /// must reproduce it or that row is dropped (fail-closed — a walk/convention
    /// mismatch must not steer α). Returns `None` if no row validated. Stable
    /// neurons are masked to 0 (via `pre_lowers[r][i]==0`), matching the local
    /// path so channel-only α reduction is not corrupted by spurious gradients.
    ///
    /// Rows run in parallel (rayon); the inner conv GEMMs degrade to `Par::Seq`
    /// under the rayon workers. `NY_ROOT_ALPHA_TRUE_MAXROWS=k` caps the summed
    /// rows for throughput measurement (default: all output rows).
    #[allow(clippy::too_many_arguments)]
    fn true_root_warmup_gradients(
        &self,
        segments: &[GpuResnetSegment],
        relu_names: &[String],
        pre_lowers: &[Vec<f32>],
        in_lo: &[f32],
        in_hi: &[f32],
        od: usize,
        gpu_lbs: &[f32],
        probe: bool,
    ) -> Option<Vec<Vec<f32>>> {
        use rayon::prelude::*;
        let n_relu = relu_names.len();
        if od == 0 || gpu_lbs.len() < od {
            return None;
        }
        let max_rows = std::env::var("NY_ROOT_ALPHA_TRUE_MAXROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        let n_rows = od.min(max_rows).max(1).min(od);

        // Per-output-row true gradient (fold order), replayed in parallel.
        let per_row: Vec<Option<Vec<Vec<f32>>>> = (0..n_rows)
            .into_par_iter()
            .map(|r| {
                let gpu_lb = gpu_lbs[r];
                if !gpu_lb.is_finite() {
                    return None;
                }
                let mut spec = vec![0.0f32; od];
                spec[r] = 1.0;
                crate::beta_crown::engine::graph::propagation::batched::wide_alpha_true::true_alpha_grads_for_row(
                    segments, &spec, &[], in_lo, in_hi, n_relu, gpu_lb, false,
                )
            })
            .collect();

        // Sum the per-row gradients (each per-neuron, consistent shape across rows).
        let mut acc: Option<Vec<Vec<f32>>> = None;
        let mut ok = 0usize;
        for row in per_row.into_iter().flatten() {
            if row.len() != n_relu {
                continue;
            }
            match acc.as_mut() {
                None => acc = Some(row),
                Some(a) => {
                    for (ar, rr) in a.iter_mut().zip(row.iter()) {
                        if ar.len() != rr.len() {
                            continue;
                        }
                        for (x, &y) in ar.iter_mut().zip(rr.iter()) {
                            *x += y;
                        }
                    }
                }
            }
            ok += 1;
        }
        let mut acc = acc?;
        if ok == 0 {
            return None;
        }

        // Mask stable neurons (match the local rule: pre_lower==0 ⇒ no α gradient).
        for (r, g) in acc.iter_mut().enumerate() {
            if let Some(pl) = pre_lowers.get(r) {
                if pl.len() == g.len() {
                    for (gi, &plv) in g.iter_mut().zip(pl.iter()) {
                        if plv == 0.0 {
                            *gi = 0.0;
                        }
                    }
                }
            }
        }

        if probe {
            let (nz, mx) = acc.iter().flatten().fold((0usize, 0.0f32), |(nz, mx), &g| {
                (nz + usize::from(g != 0.0), mx.max(g.abs()))
            });
            eprintln!("[warmup-grad] TRUE gradient: rows_ok={ok}/{n_rows} nz={nz} max_abs={mx:e}");
        }
        Some(acc)
    }

    /// GPU-resident resnet warmup BOUND (#unsat-keystone: the #1 wall). The dag-alpha
    /// loop computes the per-iteration output bound via the CPU dag_alpha_backward_pass
    /// (~7s/iter on cifar100 → warmup eats the whole budget → BaB explores 0 domains).
    /// This computes that bound in ONE sound GPU resnet backward (identity seed over the
    /// output dim, current alpha folded). Returns `Some(output_bounds)` (flat `[od]`,
    /// element-count-compatible with the CPU path's elementwise-best tracking) or `None`
    /// → CPU fallback. Default ON, opt out `NY_RESNET_WARMUP_GPU=0`; sound (sound GPU
    /// enclosure); the gradients are filled by the GPU warmup-gradient path at the
    /// gradient site.
    pub(super) fn try_gpu_warmup_bound(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &DagAlphaRuntimeState,
    ) -> Option<BoundedTensor> {
        self.try_gpu_warmup_bound_full(ctx, node_bounds, runtime)
            .map(|(bounds, _cache)| bounds)
    }

    /// True bound-only GPU fold for the terminal alpha evaluation.
    ///
    /// Unlike [`Self::try_gpu_warmup_bound`], this calls the non-gradient GPU
    /// entry point and does not build a gradient cache that no later iteration
    /// can consume. It remains a fail-closed optional fast path.
    pub(super) fn try_gpu_warmup_bound_only(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &DagAlphaRuntimeState,
    ) -> Option<BoundedTensor> {
        if !resnet_warmup_gpu_enabled() {
            return None;
        }
        let gpu = ctx
            .engine
            .and_then(|e| e.as_gpu_crown_backward())
            .filter(|g| g.provides_sound_gpu_crown())?;
        let output_node = if self.output_node.is_empty() {
            ctx.exec_order.last()?.clone()
        } else {
            self.output_node.clone()
        };
        let (segments, _relu_names, frontier_abs, node_abs) =
            self.warmup_segments(ctx, node_bounds, runtime, &output_node)?;
        let od = ctx.output_dim;
        let seed = identity_warmup_seed(od)?;
        let in_lo: Vec<f32> = ctx.input.lower().iter().copied().collect();
        let in_hi: Vec<f32> = ctx.input.upper().iter().copied().collect();
        let result = gpu
            .crown_backward_gpu_resnet_sound(
                &segments,
                &seed,
                &in_lo,
                &in_hi,
                &frontier_abs,
                &node_abs,
            )
            .ok()?;
        if result.lower_bounds.len() != od
            || result.upper_bounds.len() != od
            || result
                .lower_bounds
                .iter()
                .chain(result.upper_bounds.iter())
                .any(|v| !v.is_finite())
        {
            return None;
        }
        let lower =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[od]), result.lower_bounds).ok()?;
        let upper =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[od]), result.upper_bounds).ok()?;
        BoundedTensor::new_repaired(lower, upper, ny_tensor::RepairStrategy::Widen).ok()
    }

    /// #root-alpha-gpu (B): full-fold variant of [`Self::try_gpu_warmup_bound`]
    /// — the same single sound GPU resnet backward, returning BOTH the output
    /// bound and a [`WarmupGpuIterCache`] carrying the fold's segments /
    /// fold-order relu names / masked pre-lowers / local-rule α gradients /
    /// per-row lower bounds, so the in-loop gradient site can consume the same
    /// kernel call's outputs instead of a second extraction + kernel run. The
    /// caller stamps `iter`; `refresh_fired` starts `false`. The bound value is
    /// byte-identical to the bound-only wrapper (same body; the cache is built
    /// from outputs the wrapper discards).
    pub(super) fn try_gpu_warmup_bound_full(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &DagAlphaRuntimeState,
    ) -> Option<(BoundedTensor, WarmupGpuIterCache)> {
        if !resnet_warmup_gpu_enabled() {
            return None;
        }
        // NOTE: the `Analytic` gradient-method guard lives at the IN-LOOP call site
        // (propagate_dag/mod.rs): Analytic takes its per-ReLU gradients from the CPU
        // backward's in-place fill, so the in-loop backward must stay on CPU there.
        // The PRE-LOOP initial-CROWN call site fills no gradients, so it may take
        // the GPU bound under any gradient method (#w4-gpu-dag-backward).
        let gpu = ctx
            .engine
            .and_then(|e| e.as_gpu_crown_backward())
            .filter(|g| g.provides_sound_gpu_crown())?;
        let output_node = if self.output_node.is_empty() {
            ctx.exec_order.last()?.clone()
        } else {
            self.output_node.clone()
        };
        let input = ctx.input;
        // #root-alpha-gpu (A): skeleton fold first, legacy extraction fallback.
        let (segments, relu_names, frontier_abs, node_abs) =
            self.warmup_segments(ctx, node_bounds, runtime, &output_node)?;
        let od = ctx.output_dim;
        let seed = identity_warmup_seed(od)?;
        // Masked pre-activation lower per ReLU in FOLD order (mirrors the gradient path).
        let pre_lowers =
            self.warmup_masked_pre_lowers(ctx, node_bounds, runtime.graph(), &relu_names)?;
        let in_lo: Vec<f32> = input.lower().iter().copied().collect();
        let in_hi: Vec<f32> = input.upper().iter().copied().collect();
        let result = gpu
            .crown_backward_gpu_resnet_sound_grad(
                &segments,
                &seed,
                &in_lo,
                &in_hi,
                &pre_lowers,
                &frontier_abs,
                &node_abs,
            )
            .ok()?;
        if result.lower_bounds.len() != od
            || result.upper_bounds.len() != od
            || result
                .lower_bounds
                .iter()
                .chain(result.upper_bounds.iter())
                .any(|v| !v.is_finite())
        {
            return None;
        }
        if std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1") {
            let mn = result
                .lower_bounds
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            let mx = result
                .lower_bounds
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[warmup-bound] GPU SUCCESS od={od} lower_min={mn:.4e} lower_max={mx:.4e}");
        }
        let per_row_lower = result.lower_bounds.clone();
        let lower =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[od]), result.lower_bounds).ok()?;
        let upper =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[od]), result.upper_bounds).ok()?;
        let bounds =
            BoundedTensor::new_repaired(lower, upper, ny_tensor::RepairStrategy::Widen).ok()?;
        let cache = WarmupGpuIterCache {
            // Stamped by the owning loop; 0 here so an unstamped cache can only
            // match iteration 0 of the loop that produced it.
            iter: 0,
            segments,
            relu_names,
            pre_lowers,
            relu_grads: result.relu_grads,
            lower_bounds: per_row_lower,
            refresh_fired: false,
        };
        Some((bounds, cache))
    }

    /// #root-alpha-gpu (A): the shared warmup extraction — skeleton fold first,
    /// legacy walk as the fail-closed fallback.
    ///
    /// With `NY_ROOT_ALPHA_GPU=1` the loop builds a
    /// [`crate::network::graph_alpha::resnet_skeleton::ResnetSegmentSkeleton`]
    /// once (propagate_dag/mod.rs) and every per-iteration warmup site re-folds
    /// ONLY the per-domain slots here — static `Arc` payloads stay shared
    /// across folds, so the resident backend's weight buffers hit by pointer
    /// identity. ANY miss — no skeleton (gate off / build refusal), stale
    /// graph, or per-domain fold refusal — falls through to the legacy
    /// `extract_gpu_resnet_segments_with_relu_names`, the exact
    /// `prep_resnet_domain_with` fail-closed pattern. The fold is
    /// oracle-proven bit-identical to the legacy walk whenever both succeed
    /// (resnet_skeleton tests + the warmup-site parity oracle below), so this
    /// introduces no new bound channel.
    fn warmup_segments(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &DagAlphaRuntimeState,
        output_node: &str,
    ) -> Option<(
        Vec<GpuResnetSegment>,
        Vec<String>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
    )> {
        if let Some(skeleton) = runtime.warmup_skeleton() {
            if skeleton.matches_graph(self) {
                if let Some(folded) = skeleton.fold_for_domain(
                    self,
                    ctx.input,
                    node_bounds,
                    node_bounds,
                    Some(runtime.graph()),
                ) {
                    return Some(folded);
                }
            }
        }
        extract_gpu_resnet_segments_with_relu_names(
            self,
            ctx.input,
            output_node,
            node_bounds,
            node_bounds,
            Some(runtime.graph()),
        )
    }

    /// Masked pre-activation lower bound per ReLU in FOLD order (mirrors
    /// `extract_node_layer`'s mask handling, incl. #4404 channel→spatial
    /// expansion). Shared by the warmup bound and gradient folds — previously
    /// duplicated at both sites.
    fn warmup_masked_pre_lowers(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        relu_names: &[String],
    ) -> Option<Vec<Vec<f32>>> {
        let input = ctx.input;
        let mut pre_lowers: Vec<Vec<f32>> = Vec::with_capacity(relu_names.len());
        for name in relu_names {
            let node = self.nodes.get(name)?;
            let input_name = node.inputs.first()?;
            let pre = if input_name == NETWORK_INPUT {
                input
            } else {
                node_bounds.get(input_name)?
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
        Some(pre_lowers)
    }
}

/// Identity seed over the output dim (the warmup optimizes the output bounds),
/// matching the CPU backward's seeding. `None` guards a degenerate or
/// oversized objective/seed — shared by the warmup bound and gradient folds.
fn identity_warmup_seed(od: usize) -> Option<GpuCrownSeed> {
    if od == 0 || od > 512 || od.saturating_mul(od) > (1 << 24) {
        return None;
    }
    let mut seed_a = vec![0.0f32; od * od];
    for r in 0..od {
        seed_a[r * od + r] = 1.0;
    }
    Some(GpuCrownSeed {
        lower_a: seed_a.clone().into(),
        upper_a: seed_a.into(),
        lower_b: vec![0.0f32; od].into(),
        upper_b: vec![0.0f32; od].into(),
        num_specs: od,
        current_dim: od,
    })
}

// #root-alpha-gpu oracles: (a) gate-off byte-identity, (b) skeleton-fold vs
// legacy parity at the warmup site, (c) cache-reuse bitwise parity + one fold
// per iteration, (d) refresh/iter invalidation.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::AlphaCrownConfig;
    use crate::network::graph_alpha::resnet_skeleton::build_resnet_segment_skeleton;
    use crate::network::graph_alpha::resnet_skeleton::test_support::{
        assert_extraction_bits_eq, box_input, conv_resnet_fixture, mk_alpha, relu, static_arc_ptrs,
        CONV_FIXTURE_RELUS,
    };
    use ny_core::{GpuCrownGradResult, GpuCrownLayer, GpuCrownResult, NaiveCpuGemmEngine, NyError};
    use ny_test_utils::env::with_env_edits;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct WarmupFixture {
        graph: GraphNetwork,
        input: BoundedTensor,
        bounds: HashMap<String, BoundedTensor>,
        exec_order: Vec<String>,
        relu_nodes: Vec<(String, usize)>,
        output_dim: usize,
        input_dim: usize,
        config: AlphaCrownConfig,
    }

    /// The resnet_skeleton conv fixture, prepared for the dag-alpha warmup call
    /// sites (ctx geometry + per-ReLU neuron counts).
    fn warmup_fixture() -> WarmupFixture {
        let graph = conv_resnet_fixture();
        let input = box_input(&[2, 6, 6], -1.0, 1.0);
        let bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let exec_order: Vec<String> = graph.node_order.clone();
        let relu_nodes: Vec<(String, usize)> = CONV_FIXTURE_RELUS
            .iter()
            .map(|name| {
                let pre = graph
                    .nodes
                    .get(*name)
                    .expect("fixture relu node")
                    .inputs
                    .first()
                    .expect("relu input")
                    .clone();
                let len = bounds.get(&pre).expect("pre bounds").lower().len();
                ((*name).to_string(), len)
            })
            .collect();
        let output_dim = bounds.get("conv_out").expect("output bounds").lower().len();
        let input_dim = input.lower().len();
        WarmupFixture {
            graph,
            input,
            bounds,
            exec_order,
            relu_nodes,
            output_dim,
            input_dim,
            config: AlphaCrownConfig::default(),
        }
    }

    fn runtime_for(fix: &WarmupFixture) -> DagAlphaRuntimeState {
        let alpha = mk_alpha(&fix.graph, &fix.bounds, &CONV_FIXTURE_RELUS, 0.35, 0.65);
        DagAlphaRuntimeState::new(
            alpha,
            None,
            CONV_FIXTURE_RELUS.iter().map(|s| s.to_string()).collect(),
        )
    }

    fn ctx_of<'a>(
        fix: &'a WarmupFixture,
        engine: Option<&'a dyn ny_core::GemmEngine>,
    ) -> DagAlphaLoopContext<'a> {
        DagAlphaLoopContext {
            input: &fix.input,
            exec_order: &fix.exec_order,
            output_dim: fix.output_dim,
            input_dim: fix.input_dim,
            config: &fix.config,
            engine,
            relu_nodes: &fix.relu_nodes,
            has_bilinear: false,
            has_mul_binary: false,
        }
    }

    fn zero_local_grads(fix: &WarmupFixture) -> Vec<Array1<f32>> {
        fix.relu_nodes
            .iter()
            .map(|(_, len)| Array1::zeros(*len))
            .collect()
    }

    fn grads_bits(grads: &[Array1<f32>]) -> Vec<Vec<u32>> {
        grads
            .iter()
            .map(|g| g.iter().map(|v| v.to_bits()).collect())
            .collect()
    }

    fn tensor_bits(bounds: &BoundedTensor) -> (Vec<usize>, Vec<u32>, Vec<u32>) {
        (
            bounds.shape().to_vec(),
            bounds.lower().iter().map(|v| v.to_bits()).collect(),
            bounds.upper().iter().map(|v| v.to_bits()).collect(),
        )
    }

    /// Deterministic CPU stand-in for the sound GPU resnet backward. The
    /// returned bounds/gradients are pure functions of the call inputs
    /// (masked pre-lowers, segment count, seed size, input box), so
    /// cache-vs-fresh comparisons are exact bitwise oracles, and the
    /// invocation counter pins "one kernel fold per iteration".
    #[derive(Default)]
    struct ScriptedResnetGradEngine {
        grad_calls: AtomicUsize,
    }

    impl ScriptedResnetGradEngine {
        fn calls(&self) -> usize {
            self.grad_calls.load(Ordering::SeqCst)
        }
    }

    impl ny_core::GemmEngine for ScriptedResnetGradEngine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }

        fn as_gpu_crown_backward(&self) -> Option<&dyn ny_core::GpuCrownBackward> {
            Some(self)
        }
    }

    impl ny_core::GpuCrownBackward for ScriptedResnetGradEngine {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp(
                "scripted engine: resnet warmup grad only".into(),
            ))
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            true
        }

        fn crown_backward_gpu_resnet_sound_grad(
            &self,
            segments: &[GpuResnetSegment],
            seed: &GpuCrownSeed,
            input_lower: &[f32],
            _input_upper: &[f32],
            relu_pre_lower: &[Vec<f32>],
            _frontier_abs: &[Vec<f32>],
            _node_abs: &[Vec<f32>],
        ) -> Result<GpuCrownGradResult> {
            self.grad_calls.fetch_add(1, Ordering::SeqCst);
            let od = seed.num_specs;
            let seg_fp = segments.len() as f32 * 0.125;
            let in_fp = input_lower.first().copied().unwrap_or(0.0) * 0.25;
            let relu_grads = relu_pre_lower
                .iter()
                .enumerate()
                .map(|(r, pl)| {
                    pl.iter()
                        .enumerate()
                        .map(|(i, &l)| l * 0.5 + seg_fp + in_fp + r as f32 * 1e-3 + i as f32 * 1e-4)
                        .collect()
                })
                .collect();
            Ok(GpuCrownGradResult {
                lower_bounds: (0..od).map(|r| -1.0 - r as f32 * 0.01 + seg_fp).collect(),
                upper_bounds: (0..od).map(|r| 1.0 + r as f32 * 0.01 + seg_fp).collect(),
                relu_grads,
            })
        }
    }

    /// Gate parse: `NY_ROOT_ALPHA_GPU` is default OFF and only `"1"` enables.
    #[test]
    fn root_alpha_gpu_gate_default_off() {
        with_env_edits(|env| {
            env.remove("NY_ROOT_ALPHA_GPU");
            assert!(!root_alpha_gpu_enabled(), "unset must be OFF");
            env.set("NY_ROOT_ALPHA_GPU", "0");
            assert!(!root_alpha_gpu_enabled(), "\"0\" must be OFF");
            env.set("NY_ROOT_ALPHA_GPU", "true");
            assert!(!root_alpha_gpu_enabled(), "non-\"1\" must be OFF");
            env.set("NY_ROOT_ALPHA_GPU", "1");
            assert!(root_alpha_gpu_enabled(), "\"1\" must be ON");
        });
    }

    /// Oracle (b): `warmup_segments` with a built skeleton returns the
    /// bit-identical `(segments, relu_names, frontier_abs, node_abs)` tuple as
    /// the legacy extraction, and consecutive folds share every static `Arc`
    /// payload (the proof the skeleton path — not the legacy walk — fired).
    #[test]
    fn warmup_segments_skeleton_fold_bit_identical_to_legacy() {
        let fix = warmup_fixture();
        let mut runtime = runtime_for(&fix);
        let skeleton = build_resnet_segment_skeleton(
            &fix.graph,
            &fix.input,
            "conv_out",
            &fix.bounds,
            &fix.bounds,
            Some(runtime.graph()),
            /*allow_pure_chain=*/ false,
        )
        .expect("skeleton builds on the conv resnet fixture");
        runtime.set_warmup_skeleton(Some(skeleton));
        let ctx = ctx_of(&fix, None);

        let via_skeleton = fix
            .graph
            .warmup_segments(&ctx, &fix.bounds, &runtime, "conv_out")
            .expect("skeleton fold path succeeds");
        let legacy = extract_gpu_resnet_segments_with_relu_names(
            &fix.graph,
            &fix.input,
            "conv_out",
            &fix.bounds,
            &fix.bounds,
            Some(runtime.graph()),
        )
        .expect("legacy extraction succeeds");
        assert_extraction_bits_eq(&via_skeleton, &legacy, "warmup_segments skeleton-vs-legacy");

        let again = fix
            .graph
            .warmup_segments(&ctx, &fix.bounds, &runtime, "conv_out")
            .expect("second fold succeeds");
        let p1 = static_arc_ptrs(&via_skeleton.0);
        assert!(!p1.is_empty(), "fixture has static Arc payloads");
        assert_eq!(
            p1,
            static_arc_ptrs(&again.0),
            "consecutive skeleton folds share static Arcs (the re-fold payoff)"
        );
        assert_ne!(
            p1,
            static_arc_ptrs(&legacy.0),
            "the legacy conv walk re-materializes payloads per extraction"
        );
    }

    /// Oracle (a) shape + refusal agreement: with NO skeleton (the
    /// NY_ROOT_ALPHA_GPU default — the field is `None`) and with a STALE
    /// skeleton (built from a different graph), the helper lands on the legacy
    /// extraction with bit-identical results — never a divergent segment list.
    #[test]
    fn warmup_segments_without_or_with_stale_skeleton_takes_legacy_branch() {
        let fix = warmup_fixture();
        let mut runtime = runtime_for(&fix);
        let ctx = ctx_of(&fix, None);
        let legacy = extract_gpu_resnet_segments_with_relu_names(
            &fix.graph,
            &fix.input,
            "conv_out",
            &fix.bounds,
            &fix.bounds,
            Some(runtime.graph()),
        )
        .expect("legacy extraction succeeds");

        let no_skeleton = fix
            .graph
            .warmup_segments(&ctx, &fix.bounds, &runtime, "conv_out")
            .expect("legacy branch succeeds");
        assert_extraction_bits_eq(&no_skeleton, &legacy, "no-skeleton = legacy");

        let mut extended = conv_resnet_fixture();
        extended.add_node(relu("extra", "conv_out"));
        let ext_bounds = extended
            .collect_node_bounds(&fix.input)
            .expect("extended bounds");
        let stale = build_resnet_segment_skeleton(
            &extended,
            &fix.input,
            "conv_out",
            &ext_bounds,
            &ext_bounds,
            None,
            false,
        )
        .expect("stale skeleton builds against the extended graph");
        assert!(!stale.matches_graph(&fix.graph), "fixture must be stale");
        runtime.set_warmup_skeleton(Some(stale));
        let via_stale = fix
            .graph
            .warmup_segments(&ctx, &fix.bounds, &runtime, "conv_out")
            .expect("stale skeleton falls back to legacy");
        assert_extraction_bits_eq(&via_stale, &legacy, "stale-skeleton fallback = legacy");
    }

    /// Oracle (c): gradients consumed from the iteration cache are BITWISE
    /// equal to a fresh extraction+kernel fold at the same (α, bounds) state,
    /// the cache hit runs NO new kernel invocation, and the bound-only wrapper
    /// is bitwise-identical to the full fold's bound. Determinism note: the
    /// scripted kernel is deterministic, making cache-vs-fresh exact; against
    /// a nondeterministic kernel the cache path still returns the loop-top
    /// kernel's CAPTURED output verbatim (it never re-invokes), which this
    /// test pins via the second cache-path run.
    #[test]
    fn warmup_gradient_cache_reuse_bitwise_matches_fresh_fold() {
        with_env_edits(|env| {
            env.set("NY_ROOT_ALPHA_GPU", "1");
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_EXTRACT_SKELETON");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA_GPU");
            env.remove("NY_ROOT_ALPHA_TRUE");
            env.remove("NY_BETA_GPU_PROBE");

            let fix = warmup_fixture();
            let mut runtime = runtime_for(&fix);
            let skeleton = build_resnet_segment_skeleton(
                &fix.graph,
                &fix.input,
                "conv_out",
                &fix.bounds,
                &fix.bounds,
                Some(runtime.graph()),
                false,
            )
            .expect("skeleton builds");
            runtime.set_warmup_skeleton(Some(skeleton));
            let engine = ScriptedResnetGradEngine::default();
            let ctx = ctx_of(&fix, Some(&engine));
            let local_grads = zero_local_grads(&fix);

            // Loop-top full fold: ONE kernel call producing bound + cache.
            let (bound_full, mut cache) = fix
                .graph
                .try_gpu_warmup_bound_full(&ctx, &fix.bounds, &runtime)
                .expect("full warmup fold succeeds");
            assert_eq!(engine.calls(), 1, "full fold = one kernel call");
            cache.iter = 4;

            // Bound-only wrapper parity (the pre-loop site): same body ⇒
            // bitwise-identical bound.
            let bound_wrapper = fix
                .graph
                .try_gpu_warmup_bound(&ctx, &fix.bounds, &runtime)
                .expect("wrapper succeeds");
            assert_eq!(
                tensor_bits(&bound_full),
                tensor_bits(&bound_wrapper),
                "wrapper bound must be bitwise-identical to the full fold's"
            );

            // Fresh gradient-site fold (no cache): its own kernel call.
            let fresh = fix
                .graph
                .try_gpu_resnet_warmup_gradients(&ctx, &fix.bounds, &runtime, &local_grads, None, 4)
                .expect("fresh gradient fold succeeds");
            let calls_after_fresh = engine.calls();
            assert_eq!(
                calls_after_fresh, 3,
                "fresh gradient fold re-runs the kernel"
            );
            assert!(
                fresh.iter().flatten().any(|&g| g != 0.0),
                "scripted grads must be nonzero for the parity check to bite"
            );

            // Cache hit: NO new kernel call, bitwise-identical grads.
            let cached = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_grads,
                    Some(&cache),
                    4,
                )
                .expect("cached gradient path succeeds");
            assert_eq!(
                engine.calls(),
                calls_after_fresh,
                "cache hit must not re-run the kernel (one fold per iteration)"
            );
            assert_eq!(
                grads_bits(&fresh),
                grads_bits(&cached),
                "cache-path grads must be bitwise equal to the fresh fold"
            );

            // Second cache-path run: bitwise-stable (a pure CPU mapping of the
            // captured kernel output — no re-invocation).
            let cached_again = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_grads,
                    Some(&cache),
                    4,
                )
                .expect("second cached run succeeds");
            assert_eq!(engine.calls(), calls_after_fresh);
            assert_eq!(grads_bits(&cached), grads_bits(&cached_again));
        });
    }

    /// Oracle (d) + gate-off byte-identity at the consumption site: a
    /// sentinel-poisoned cache is consumed ONLY when the gate is on, the iter
    /// matches, and no refresh fired; `refresh_fired`, a stale iter, or the
    /// gate being off each force a fresh kernel fold with the un-poisoned
    /// results.
    #[test]
    fn warmup_gradient_cache_invalidation_refresh_iter_and_gate() {
        with_env_edits(|env| {
            env.set("NY_ROOT_ALPHA_GPU", "1");
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_EXTRACT_SKELETON");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA_GPU");
            env.remove("NY_ROOT_ALPHA_TRUE");
            env.remove("NY_BETA_GPU_PROBE");

            let fix = warmup_fixture();
            let runtime = runtime_for(&fix);
            let engine = ScriptedResnetGradEngine::default();
            let ctx = ctx_of(&fix, Some(&engine));
            let local_grads = zero_local_grads(&fix);

            let (_bound, mut cache) = fix
                .graph
                .try_gpu_warmup_bound_full(&ctx, &fix.bounds, &runtime)
                .expect("full warmup fold succeeds");
            cache.iter = 2;
            // Sentinel-poison the cached grads so consumption is observable.
            for g in cache.relu_grads.iter_mut() {
                for v in g.iter_mut() {
                    *v = 42.0;
                }
            }

            let fresh = fix
                .graph
                .try_gpu_resnet_warmup_gradients(&ctx, &fix.bounds, &runtime, &local_grads, None, 2)
                .expect("fresh fold succeeds");

            // Fresh cache consumed: every relu is covered by the fold, so the
            // sentinel must appear everywhere.
            let hit = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_grads,
                    Some(&cache),
                    2,
                )
                .expect("cache hit succeeds");
            assert!(
                hit.iter().flatten().all(|&g| g == 42.0),
                "a fresh cache must be consumed (sentinel grads returned)"
            );
            let calls_before = engine.calls();

            // refresh_fired ⇒ fresh fold that iteration (oracle d).
            cache.refresh_fired = true;
            let after_refresh = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_grads,
                    Some(&cache),
                    2,
                )
                .expect("refresh-invalidated path succeeds");
            assert_eq!(
                engine.calls(),
                calls_before + 1,
                "refresh_fired must force a fresh kernel fold"
            );
            assert_eq!(grads_bits(&after_refresh), grads_bits(&fresh));
            cache.refresh_fired = false;

            // Stale iter ⇒ fresh fold.
            let stale_iter = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_grads,
                    Some(&cache),
                    3,
                )
                .expect("stale-iter path succeeds");
            assert_eq!(
                engine.calls(),
                calls_before + 2,
                "a stale iter must force a fresh kernel fold"
            );
            assert_eq!(grads_bits(&stale_iter), grads_bits(&fresh));

            // Gate off ⇒ the cache is ignored entirely (oracle a).
            env.remove("NY_ROOT_ALPHA_GPU");
            let gate_off = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_grads,
                    Some(&cache),
                    2,
                )
                .expect("gate-off path succeeds");
            assert_eq!(
                engine.calls(),
                calls_before + 3,
                "with the gate off the cache must be ignored"
            );
            assert_eq!(grads_bits(&gate_off), grads_bits(&fresh));
        });
    }

    /// Loop-level gate oracle: the full dag-alpha optimization loop returns
    /// BITWISE-identical bounds with NY_ROOT_ALPHA_GPU=1 vs unset on the same
    /// fixture + deterministic engine (increments A+B change WHERE the fold
    /// runs, never its value).
    #[test]
    fn dag_alpha_loop_bit_identical_with_gate_on_and_off() {
        let run = |gate_on: bool| -> BoundedTensor {
            with_env_edits(|env| {
                if gate_on {
                    env.set("NY_ROOT_ALPHA_GPU", "1");
                } else {
                    env.remove("NY_ROOT_ALPHA_GPU");
                }
                env.set("NY_RESNET_WARMUP_GPU", "1");
                env.remove("NY_EXTRACT_SKELETON");
                env.remove("NY_MULTIOBJ_JOINT_ALPHA");
                env.remove("NY_MULTIOBJ_JOINT_ALPHA_GPU");
                env.remove("NY_ROOT_ALPHA_TRUE");
                env.remove("NY_BETA_GPU_PROBE");

                let fix = warmup_fixture();
                let engine = ScriptedResnetGradEngine::default();
                let config = AlphaCrownConfig {
                    iterations: 3,
                    ..Default::default()
                };
                fix.graph
                    .propagate_dag_alpha_crown_with_config_and_engine(
                        &fix.input,
                        &config,
                        Some(&engine),
                    )
                    .expect("dag alpha loop runs")
            })
        };
        let on = run(true);
        let off = run(false);
        assert_eq!(
            tensor_bits(&on),
            tensor_bits(&off),
            "gate on/off must be bitwise-identical end to end"
        );
    }
}
