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
use std::time::Instant;
use tracing::{debug, info};

use super::super::runtime_state::DagAlphaRuntimeState;
use super::DagAlphaLoopContext;
use crate::network::core::{GraphNetwork, NETWORK_INPUT};
use crate::network::graph_alpha::resnet_decompose::{
    extract_gpu_resnet_segments_with_relu_names, joint_alpha_grads_fold,
    joint_alpha_grads_fold_gpu_with_deadline, multiobj_joint_alpha_enabled,
    multiobj_joint_alpha_gpu_enabled, resnet_warmup_gpu_enabled, root_alpha_gpu_enabled,
    DeadlineJointAlphaFoldError,
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
    /// Whether the subordinate margin-gradient lane actually dispatched its
    /// resident joint adjoint. The owning loop uses this outcome, rather than
    /// mere gate eligibility, for telemetry and objective-specific stopping.
    pub(super) margin_dispatch: MarginGradientDispatch,
}

/// Per-iteration request from the root margin-gradient lane.
///
/// `NoBinding` is distinct from `Disabled`: an armed/eligible lane with no
/// finite unresolved row must take the bounded local-gradient fallback and
/// must never enter the legacy unbounded AnalyticChain replay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum MarginGradientRequest<'a> {
    Disabled,
    NoBinding,
    Binding(&'a [f32]),
}

impl MarginGradientRequest<'_> {
    fn is_armed(self) -> bool {
        !matches!(self, Self::Disabled)
    }

    fn objective_ref(&self) -> Option<&[f32]> {
        match self {
            Self::Binding(objective) => Some(*objective),
            Self::Disabled | Self::NoBinding => None,
        }
    }
}

/// Whether the historical post-fold deadline guard must refuse this result.
///
/// The pre-feature cache-hit path never performed this refusal. Preserve that
/// exact behavior when the child lane is disabled, while an armed child must
/// always honor its bounded deadline even when it reused a resident cache.
fn post_cache_deadline_refusal_required(
    margin_armed: bool,
    used_cache: bool,
    deadline_expired: bool,
) -> bool {
    deadline_expired && (margin_armed || !used_cache)
}

fn margin_deadline_check(
    deadline: Instant,
) -> std::result::Result<(), DeadlineJointAlphaFoldError> {
    if Instant::now() >= deadline {
        Err(DeadlineJointAlphaFoldError::DeadlineExpired)
    } else {
        Ok(())
    }
}

fn margin_deadline_host_work(
    deadline: Instant,
    completed: &mut usize,
) -> std::result::Result<(), DeadlineJointAlphaFoldError> {
    if completed.is_multiple_of(4096) {
        margin_deadline_check(deadline)?;
    }
    *completed = completed
        .checked_add(1)
        .ok_or(DeadlineJointAlphaFoldError::MappingMismatch)?;
    Ok(())
}

/// Why an armed margin-gradient iteration declined the resident joint adjoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MarginGradientFallbackReason {
    NoBinding,
    DeadlineExpired,
    ResidentUnavailable,
    InvalidObjective,
    JointUnavailable,
    NonFiniteGradient,
    MappingMismatch,
}

impl MarginGradientFallbackReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::NoBinding => "no_binding",
            Self::DeadlineExpired => "deadline_expired",
            Self::ResidentUnavailable => "resident_unavailable",
            Self::InvalidObjective => "invalid_objective",
            Self::JointUnavailable => "joint_unavailable",
            Self::NonFiniteGradient => "nonfinite_gradient",
            Self::MappingMismatch => "mapping_mismatch",
        }
    }
}

fn deadline_joint_fold_fallback_reason(
    error: DeadlineJointAlphaFoldError,
) -> MarginGradientFallbackReason {
    match error {
        DeadlineJointAlphaFoldError::DeadlineExpired => {
            MarginGradientFallbackReason::DeadlineExpired
        }
        DeadlineJointAlphaFoldError::JointUnavailable => {
            MarginGradientFallbackReason::JointUnavailable
        }
        DeadlineJointAlphaFoldError::NonFiniteGradient => {
            MarginGradientFallbackReason::NonFiniteGradient
        }
        DeadlineJointAlphaFoldError::MappingMismatch => {
            MarginGradientFallbackReason::MappingMismatch
        }
    }
}

/// Source of the bounded gradient used after an armed resident-joint refusal.
///
/// `ResidentLocalRule` is parity with the child-disabled resident warmup path.
/// `SuppliedLocal` is the pre-resident/unmappable moat supplied by the caller;
/// it is still bounded and soundness-neutral, but must not be reported as
/// resident-local parity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MarginGradientFallbackSource {
    ResidentLocalRule,
    SuppliedLocal,
}

impl MarginGradientFallbackSource {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::ResidentLocalRule => "resident_local_rule",
            Self::SuppliedLocal => "supplied_local",
        }
    }
}

/// Which engine class actually computed a dispatched joint adjoint.
///
/// `Resident` is the verdict-authority backend (`provides_sound_gpu_crown`)
/// — the pre-existing CUDA lane, whose telemetry stays byte-identical.
/// `WgpuProposal` is the dedicated α-steering proposal channel
/// (#alpha-steering-proposal): a proposal-grade engine with NO verdict
/// authority whose adjoint output is consumed EXCLUSIVELY as margin-gradient
/// input; it sits beside `supplied_local`/`resident_local_rule` in the
/// dispatch log line and the flight record.
/// `CpuReplay` is the deterministic f64 binding-row replay
/// (#binding-row-replay, `binding_row_true_alpha_grads`): the FD-proven true
/// d(binding-row)/dα of the certified CPU fold itself, dispatched on
/// non-resident hosts when the proposal channel is absent or typed-refuses.
/// Like the proposal channel it produces gradients only — it can never touch
/// a bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MarginGradientJointSource {
    Resident,
    WgpuProposal,
    CpuReplay,
}

impl MarginGradientJointSource {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::WgpuProposal => "wgpu_proposal",
            Self::CpuReplay => "cpu_replay",
        }
    }
}

/// Auditable outcome of the subordinate margin-gradient request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MarginGradientDispatch {
    NotRequested,
    JointDispatched {
        source: MarginGradientJointSource,
    },
    LocalFallback {
        reason: MarginGradientFallbackReason,
        source: MarginGradientFallbackSource,
    },
}

impl MarginGradientDispatch {
    pub(super) fn joint_dispatched(self) -> bool {
        matches!(self, Self::JointDispatched { .. })
    }
}

struct WarmupGradientResult {
    gradients: Vec<Array1<f32>>,
    margin_dispatch: MarginGradientDispatch,
}

impl WarmupGradientResult {
    fn legacy(gradients: Vec<Array1<f32>>) -> Self {
        Self {
            gradients,
            margin_dispatch: MarginGradientDispatch::NotRequested,
        }
    }

    fn margin_joint_resident(gradients: Vec<Array1<f32>>) -> Self {
        Self {
            gradients,
            margin_dispatch: MarginGradientDispatch::JointDispatched {
                source: MarginGradientJointSource::Resident,
            },
        }
    }

    fn margin_joint_proposal(gradients: Vec<Array1<f32>>) -> Self {
        Self {
            gradients,
            margin_dispatch: MarginGradientDispatch::JointDispatched {
                source: MarginGradientJointSource::WgpuProposal,
            },
        }
    }

    fn margin_binding_row_replay(gradients: Vec<Array1<f32>>) -> Self {
        Self {
            gradients,
            margin_dispatch: MarginGradientDispatch::JointDispatched {
                source: MarginGradientJointSource::CpuReplay,
            },
        }
    }

    fn margin_supplied_fallback(
        local_gradients: &[Array1<f32>],
        reason: MarginGradientFallbackReason,
    ) -> Self {
        Self {
            gradients: local_gradients.to_vec(),
            margin_dispatch: MarginGradientDispatch::LocalFallback {
                reason,
                source: MarginGradientFallbackSource::SuppliedLocal,
            },
        }
    }

    fn margin_resident_fallback(
        gradients: Vec<Array1<f32>>,
        reason: MarginGradientFallbackReason,
    ) -> Self {
        Self {
            gradients,
            margin_dispatch: MarginGradientDispatch::LocalFallback {
                reason,
                source: MarginGradientFallbackSource::ResidentLocalRule,
            },
        }
    }
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
pub(super) fn root_alpha_true_enabled() -> bool {
    std::env::var("NY_ROOT_ALPHA_TRUE").ok().as_deref() == Some("1")
}

thread_local! {
    /// #replay-row-index PROBE: the ascent's spec-row index for the current
    /// binding objective (see the setter's comment in `propagate_dag/mod.rs`).
    static BINDING_SEED_ROW: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

pub(super) fn set_binding_seed_row(row: Option<usize>) {
    BINDING_SEED_ROW.with(|cell| cell.set(row));
}

fn binding_seed_row() -> Option<usize> {
    BINDING_SEED_ROW.with(std::cell::Cell::get)
}

/// #replay-row-index PROBE gate (dark, default OFF ⇒ byte-identical).
fn replay_row_index_enabled() -> bool {
    std::env::var("NY_ALPHA_REPLAY_ROWIDX").ok().as_deref() == Some("1")
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
        margin_request: MarginGradientRequest<'_>,
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
        let gpu_warmup_gradients: Option<WarmupGradientResult> =
            if matches!(ctx.config.gradient_method, GradientMethod::AnalyticChain) {
                self.try_gpu_resnet_warmup_gradients(
                    ctx,
                    node_bounds,
                    runtime,
                    gradients,
                    warmup_cache,
                    iter,
                    margin_request,
                )
            } else if margin_request.is_armed() {
                // Defensive all-path moat: eligibility currently requires
                // AnalyticChain, but an armed request must still never fall
                // into another gradient method if that invariant drifts.
                Some(WarmupGradientResult::margin_supplied_fallback(
                    gradients,
                    MarginGradientFallbackReason::ResidentUnavailable,
                ))
            } else {
                None
            };
        // Compute gradient using configured method.
        //
        // NOTE: DAG α-CROWN historically ignored `config.gradient_method` and always used the
        // per-ReLU local gradients returned by `propagate_linear_with_alpha`. Honor the config
        // so the default (`AnalyticChain`) is actually used on ResNet-like graphs with skip
        // connections.
        let (numerical_gradients, margin_dispatch) = match gpu_warmup_gradients {
            Some(result) => (result.gradients, result.margin_dispatch),
            None if margin_request.is_armed() => (
                gradients.to_vec(),
                MarginGradientDispatch::LocalFallback {
                    reason: MarginGradientFallbackReason::ResidentUnavailable,
                    source: MarginGradientFallbackSource::SuppliedLocal,
                },
            ),
            None => {
                let legacy_gradients = match ctx.config.gradient_method {
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
                        // The historical replay below intentionally erases
                        // the request deadline and performs a complete
                        // intermediate/chain-gradient pass. It may create an
                        // Anchored ConvTranspose carrier, so entering it from
                        // a finite request would escape the absolute proof
                        // authority. Local analytic gradients are already
                        // available and are soundness-neutral steering data;
                        // use them until the replay itself is cooperative.
                        if ctx.config.deadline.is_some() {
                            gradients.to_vec()
                        } else {
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
                                    let chain_grads = self
                                        .compute_graph_chain_rule_gradients_with_binding(
                                            ctx.input,
                                            &relu_names,
                                            &intermediate,
                                            Some(runtime.graph()),
                                            ctx.engine,
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
                    }
                };
                (legacy_gradients, MarginGradientDispatch::NotRequested)
            }
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
            margin_dispatch,
        })
    }

    /// Map fold-order gradients into the runtime ReLU order.
    ///
    /// The supplied gradients seed ReLUs outside the decomposed suffix. `None`
    /// means the resident source cannot be mapped without guessing, so an
    /// armed caller must use and report the supplied-local moat instead.
    ///
    /// #channel-alpha-grad: when the walk's α for a conv node is
    /// channel-shared (length C, `full_conv_alpha: false` presets) the
    /// per-neuron fold gradient (length C·H·W) is reduced by the exact chain
    /// rule `dL/dα_c = Σ_{h,w} dL/dα_{c,h,w}` BEFORE the length check — keyed
    /// on MEASURED shapes only ([`GraphAlphaState::channel_reduction_geometry`]),
    /// never a config flag. Genuinely irreconcilable layouts still refuse.
    fn map_warmup_fold_gradients(
        ctx: &DagAlphaLoopContext<'_>,
        alpha_state: &GraphAlphaState,
        supplied_gradients: &[Array1<f32>],
        relu_names: &[String],
        fold_gradients: &[Vec<f32>],
    ) -> Option<Vec<Array1<f32>>> {
        if relu_names.len() != fold_gradients.len() {
            return None;
        }
        let pos: HashMap<&str, usize> = ctx
            .relu_nodes
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.as_str(), i))
            .collect();
        let mut out = supplied_gradients.to_vec();
        for (name, gradient) in relu_names.iter().zip(fold_gradients) {
            let &index = pos.get(name.as_str())?;
            let target_len = out.get(index)?.len();
            if gradient.len() == target_len {
                out[index] = Array1::from(gradient.clone());
            } else if let Some((channels, spatial)) =
                alpha_state.channel_reduction_geometry(name, target_len, gradient.len())
            {
                let mut reduced = vec![0.0f32; channels];
                for (c, slot) in reduced.iter_mut().enumerate() {
                    for s in 0..spatial {
                        *slot += gradient[c * spatial + s];
                    }
                }
                out[index] = Array1::from(reduced);
            } else {
                return None;
            }
        }
        Some(out)
    }

    /// Deadline-polled mapping twin used only by the armed margin child.
    ///
    /// Keeping this separate leaves the feature-disabled clone/mapping path
    /// byte-for-byte untouched while preventing a timely backend result from
    /// acquiring an unbounded host-side mapping tail.
    ///
    /// #channel-alpha-grad: this is the seam BOTH true-gradient lanes pass
    /// through (CUDA narrow joint branch, resident joint fold, wgpu proposal
    /// channel — so GB10 inherits it too). Channel-shared α (length C) with a
    /// per-neuron fold gradient (length C·H·W) reduces by the channel sum
    /// `dL/dα_c = Σ_{h,w} dL/dα_{c,h,w}` before the length check, keyed on
    /// MEASURED shapes ([`GraphAlphaState::channel_reduction_geometry`]);
    /// irreconcilable layouts keep the `Ok(None)` refusal.
    fn map_warmup_fold_gradients_with_deadline(
        ctx: &DagAlphaLoopContext<'_>,
        alpha_state: &GraphAlphaState,
        supplied_gradients: &[Array1<f32>],
        relu_names: &[String],
        fold_gradients: &[Vec<f32>],
        deadline: Instant,
    ) -> std::result::Result<Option<Vec<Array1<f32>>>, DeadlineJointAlphaFoldError> {
        margin_deadline_check(deadline)?;
        if relu_names.len() != fold_gradients.len() {
            return Ok(None);
        }

        let mut host_work = 0usize;
        let mut positions = HashMap::new();
        for (index, (name, _)) in ctx.relu_nodes.iter().enumerate() {
            margin_deadline_host_work(deadline, &mut host_work)?;
            positions.insert(name.as_str(), index);
        }
        margin_deadline_check(deadline)?;

        let mut out = Vec::new();
        out.try_reserve_exact(supplied_gradients.len())
            .map_err(|_| DeadlineJointAlphaFoldError::MappingMismatch)?;
        margin_deadline_check(deadline)?;
        for supplied in supplied_gradients {
            let mut values = Vec::new();
            values
                .try_reserve_exact(supplied.len())
                .map_err(|_| DeadlineJointAlphaFoldError::MappingMismatch)?;
            margin_deadline_check(deadline)?;
            for &value in supplied {
                margin_deadline_host_work(deadline, &mut host_work)?;
                values.push(value);
            }
            out.push(Array1::from(values));
        }

        for (name, gradient) in relu_names.iter().zip(fold_gradients) {
            margin_deadline_host_work(deadline, &mut host_work)?;
            let Some(&index) = positions.get(name.as_str()) else {
                return Ok(None);
            };
            let target_len = out.get(index).map(Array1::len).unwrap_or(usize::MAX);
            if gradient.len() == target_len {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(gradient.len())
                    .map_err(|_| DeadlineJointAlphaFoldError::MappingMismatch)?;
                margin_deadline_check(deadline)?;
                for &value in gradient {
                    margin_deadline_host_work(deadline, &mut host_work)?;
                    values.push(value);
                }
                out[index] = Array1::from(values);
            } else if let Some((channels, spatial)) =
                alpha_state.channel_reduction_geometry(name, target_len, gradient.len())
            {
                // dL/dα_c = Σ_{h,w} dL/dα_{c,h,w}: channel-shared α broadcasts
                // to every spatial position, so the true derivative is the
                // spatial sum (#channel-alpha-grad).
                let mut reduced = Vec::new();
                reduced
                    .try_reserve_exact(channels)
                    .map_err(|_| DeadlineJointAlphaFoldError::MappingMismatch)?;
                margin_deadline_check(deadline)?;
                for c in 0..channels {
                    let mut sum = 0.0f32;
                    for s in 0..spatial {
                        margin_deadline_host_work(deadline, &mut host_work)?;
                        sum += gradient[c * spatial + s];
                    }
                    reduced.push(sum);
                }
                out[index] = Array1::from(reduced);
            } else {
                return Ok(None);
            }
        }
        margin_deadline_check(deadline)?;
        Ok(Some(out))
    }

    /// GPU-resident warmup gradients for ResNet suffixes (cifar100/tinyimagenet unsat
    /// keystone, step 3b-ii). When the output suffix decomposes into GPU resnet
    /// segments, compute the per-ReLU analytic alpha gradients on the GPU in one
    /// resident backward instead of the slow CPU `dag_alpha_backward_pass_with_intermediates`
    /// (which makes the warmup overrun → 0 BaB domains at ≤400s).
    ///
    /// Without a margin request, returns `None` on any mismatch so the caller
    /// preserves the legacy CPU path. With an armed margin request, every
    /// refusal instead returns the already-computed local gradients plus an
    /// explicit fallback reason. That distinction prevents a resident miss
    /// from entering the legacy deadline-free AnalyticChain replay.
    ///
    /// Gradients are non-soundness-critical (they only steer alpha — any alpha
    /// is a sound relaxation), so even a wrong mapping yields a sound, if
    /// looser, bound; the soundness gate is never at risk here.
    #[allow(clippy::too_many_arguments)]
    fn try_gpu_resnet_warmup_gradients(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &DagAlphaRuntimeState,
        local_gradients: &[Array1<f32>],
        cache: Option<&WarmupGpuIterCache>,
        iter: usize,
        margin_request: MarginGradientRequest<'_>,
    ) -> Option<WarmupGradientResult> {
        let margin_objective = margin_request.objective_ref();
        if margin_objective.is_some_and(|objective| {
            objective.len() != ctx.output_dim || objective.iter().any(|value| !value.is_finite())
        }) {
            return Some(WarmupGradientResult::margin_supplied_fallback(
                local_gradients,
                MarginGradientFallbackReason::InvalidObjective,
            ));
        }
        let reusable_cache = if root_alpha_gpu_enabled() || margin_request.is_armed() {
            cache.filter(|candidate| {
                candidate.iter == iter
                    && !candidate.refresh_fired
                    && candidate.relu_grads.len() == candidate.relu_names.len()
            })
        } else {
            None
        };
        if margin_request.is_armed()
            && ctx
                .config
                .deadline
                .is_some_and(|value| Instant::now() >= value)
        {
            if let Some(mapped) = reusable_cache.and_then(|resident| {
                Self::map_warmup_fold_gradients(
                    ctx,
                    runtime.graph(),
                    local_gradients,
                    &resident.relu_names,
                    &resident.relu_grads,
                )
            }) {
                return Some(WarmupGradientResult::margin_resident_fallback(
                    mapped,
                    MarginGradientFallbackReason::DeadlineExpired,
                ));
            }
            return Some(WarmupGradientResult::margin_supplied_fallback(
                local_gradients,
                MarginGradientFallbackReason::DeadlineExpired,
            ));
        }

        // Keep the long-established no-margin implementation as an Option
        // attempt. The outer match is the all-path moat: any `?`/`None`
        // anywhere in extraction, resident folding, or mapping becomes a
        // bounded local fallback when the child request is armed.
        let attempt = (|| -> Option<WarmupGradientResult> {
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
            // #envelope-grad AND THE GPU LANE — CORRECTED 2026-08-17.
            //
            // This block used to `warn!` that NY_ALPHA_ENVELOPE_GRAD=1 is IGNORED
            // here. That was true when written (the resident kernel computes the
            // same sign-definite local rule `grad[i] = pre_lower[i] * sum_j
            // max(a_lower[j,i],0)`, CROWN_ALPHA_GRADIENT_SHADER) and became FALSE
            // about three hours later, when `de451dc68` landed the host-side
            // rescale `envelope_rescaled_warmup_gradients` (#envelope-grad-gpu,
            // dispatched below). The rescale is exact — `S = g/l` recovers the
            // device sum and re-multiplies by `clamp(h_hat(x*), l, u)` — and was
            // verified end-to-end: `x*` is BIT-IDENTICAL across the CPU and GPU
            // lanes (same binding_row, same sign digest), `S` is non-zero in both.
            //
            // The stale warning survived three commits telling readers to disable
            // the GPU lane to get a corrected direction they were already getting.
            // A comment that documents a fixed defect as live costs exactly what
            // the defect cost. Do not restore it.
            //
            // What is NOT settled: whether the envelope rule HELPS here. The old
            // "CPU+envelope 132 vs GPU+rescale 51" comparison was never a valid
            // A/B — the arms differ in the BOUND path as well as the gradient, and
            // even the baselines disagree (CPU local 59, GPU local 63), so the
            // interior-alpha counts are not commensurable. That question needs an
            // experiment holding the bound path fixed.
            if super::super::backward::gradients::envelope_grad_enabled() {
                static ARMED: std::sync::Once = std::sync::Once::new();
                ARMED.call_once(|| {
                    tracing::info!(
                        "NY_ALPHA_ENVELOPE_GRAD=1 with the resident GPU warmup lane: the \
                         device local-rule gradients are RESCALED into the envelope rule \
                         on the host (#envelope-grad-gpu). Confirm the \
                         `[envelope-grad-gpu] RESCALED` probe line under \
                         NY_BETA_GPU_PROBE=1 before believing any measurement through \
                         this path."
                    );
                });
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
            // Verdict-authority filter: unchanged for every consumer below —
            // resident BOUND folds and the resident joint adjoint both require
            // an engine whose numbers may decide a verdict.
            let authority_gpu = ctx
                .engine
                .and_then(|e| e.as_gpu_crown_backward())
                .filter(|g| g.provides_sound_gpu_crown());
            let Some(gpu) = authority_gpu else {
                // No verdict-authority resident backend (the Metal case — the
                // proof adapter's `as_gpu_crown_backward()` is `None` and
                // WgpuDevice's `provides_sound_gpu_crown()` is quarantined
                // `false`). An armed BINDING objective still gets a TRUE
                // d(binding-row)/dα, through a two-tier lane; the
                // wrong-direction local rule is LAST in all cases:
                //
                //   1. #alpha-steering-proposal (the ACCELERANT,
                //      arbitration doc §3): the wgpu joint adjoint. Its
                //      output is consumed exclusively as margin-gradient
                //      input (the API returns gradients only — no bound
                //      exists to leak).
                //   2. #binding-row-replay (the envelope-general PRIMARY):
                //      the deterministic CPU replay of the certified fold,
                //      when the channel is absent or typed-refuses.
                //
                // ORDER (measured, this file's own cost accounting): the
                // replay needs one `dag_alpha_backward_pass_with_intermediates`
                // per iteration — the ~27s/iter CPU AnalyticChain backward on
                // cifar100 resnet_medium (see the #w4-gpu-dag-backward note at
                // the fold-order mapping below) — while the proposal dispatch
                // completed 6 iterations in a 7s window (A/B v2, arbitration
                // doc §6). An accelerant slower than what it accelerates is a
                // contradiction, so the GPU adjoint goes first WHERE ITS
                // ENVELOPE PERMITS; the replay catches everything it refuses
                // (MaxPool/DualAlpha segments, absent adapter, no deadline,
                // non-decomposable suffixes). Cross-oracle cosine 1.000000
                // between the two (doc §2) means the order changes cost only,
                // never direction.
                //
                // Both tiers refused / disabled child ⇒ fall through to the
                // outer moat exactly as before (armed ⇒ bounded local
                // fallback; disabled ⇒ legacy CPU path, byte-identical).
                if let Some(objective) = margin_objective {
                    if let Some(result) = self.try_wgpu_proposal_joint_gradients(
                        ctx,
                        node_bounds,
                        runtime,
                        local_gradients,
                        objective,
                    ) {
                        if result.margin_dispatch.joint_dispatched() {
                            return Some(result);
                        }
                        // Typed proposal refusal → replay tier. If the replay
                        // also refuses, preserve the proposal's typed reason.
                        if let Some(replay) = self.try_binding_row_replay_gradients(
                            ctx,
                            node_bounds,
                            runtime,
                            local_gradients,
                            objective,
                        ) {
                            return Some(replay);
                        }
                        return Some(result);
                    }
                    // Channel absent (no engine / no deadline / extraction
                    // refusal) → replay tier directly.
                    if let Some(replay) = self.try_binding_row_replay_gradients(
                        ctx,
                        node_bounds,
                        runtime,
                        local_gradients,
                        objective,
                    ) {
                        return Some(replay);
                    }
                }
                return None;
            };
            let deadline = ctx.config.deadline;
            let generic_deadline_capable =
                crate::sound_gpu_gate::gpu_crown_backend_honors_deadline(gpu, deadline);
            let method_deadline_capable = margin_objective.is_some()
                && deadline.is_some()
                && gpu.provides_deadline_bounded_joint_alpha_gradient_resident();
            if !generic_deadline_capable && !method_deadline_capable {
                return None;
            }
            if deadline.is_some_and(|value| Instant::now() >= value) {
                return if margin_request.is_armed() {
                    reusable_cache
                        .and_then(|resident| {
                            Self::map_warmup_fold_gradients(
                                ctx,
                                runtime.graph(),
                                local_gradients,
                                &resident.relu_names,
                                &resident.relu_grads,
                            )
                        })
                        .map_or_else(
                            || {
                                Some(WarmupGradientResult::margin_supplied_fallback(
                                    local_gradients,
                                    MarginGradientFallbackReason::DeadlineExpired,
                                ))
                            },
                            |mapped| {
                                Some(WarmupGradientResult::margin_resident_fallback(
                                    mapped,
                                    MarginGradientFallbackReason::DeadlineExpired,
                                ))
                            },
                        )
                } else {
                    None
                };
            }

            // CUDA deliberately does not advertise the backend-global CROWN
            // deadline contract.  Its ATS joint adjoint has a narrower,
            // call-local bounded API, so an armed binding objective can still
            // extract the current segments and dispatch that exact method
            // without first entering the ordinary unbounded resident-gradient
            // fold.  On refusal there is no resident local source; preserve the
            // supplied local gradient and report it truthfully.
            if !generic_deadline_capable {
                let objective = margin_objective?;
                let deadline = deadline?;
                let output_node = if self.output_node.is_empty() {
                    ctx.exec_order.last()?.clone()
                } else {
                    self.output_node.clone()
                };
                let (segments, relu_names, _frontier_abs, _node_abs) =
                    self.warmup_segments(ctx, node_bounds, runtime, &output_node)?;
                let pre_lowers =
                    self.warmup_masked_pre_lowers(ctx, node_bounds, runtime.graph(), &relu_names)?;
                let in_lo: Vec<f32> = ctx.input.lower().iter().copied().collect();
                let in_hi: Vec<f32> = ctx.input.upper().iter().copied().collect();
                let gradients = match joint_alpha_grads_fold_gpu_with_deadline(
                    gpu,
                    &segments,
                    objective,
                    1,
                    ctx.output_dim,
                    &in_lo,
                    &in_hi,
                    &pre_lowers,
                    relu_names.len(),
                    deadline,
                ) {
                    Ok(gradients) => gradients,
                    Err(error) => {
                        return Some(WarmupGradientResult::margin_supplied_fallback(
                            local_gradients,
                            deadline_joint_fold_fallback_reason(error),
                        ));
                    }
                };
                let mapped = match Self::map_warmup_fold_gradients_with_deadline(
                    ctx,
                    runtime.graph(),
                    local_gradients,
                    &relu_names,
                    &gradients,
                    deadline,
                ) {
                    Ok(Some(mapped)) => mapped,
                    Ok(None) => {
                        return Some(WarmupGradientResult::margin_supplied_fallback(
                            local_gradients,
                            MarginGradientFallbackReason::MappingMismatch,
                        ));
                    }
                    Err(error) => {
                        return Some(WarmupGradientResult::margin_supplied_fallback(
                            local_gradients,
                            deadline_joint_fold_fallback_reason(error),
                        ));
                    }
                };
                if Instant::now() >= deadline {
                    return Some(WarmupGradientResult::margin_supplied_fallback(
                        local_gradients,
                        MarginGradientFallbackReason::DeadlineExpired,
                    ));
                }
                return Some(WarmupGradientResult::margin_joint_resident(mapped));
            }

            // #root-alpha-gpu (B): reuse the loop-top full fold's outputs instead
            // of a second extraction + kernel run in the same iteration. Valid
            // because α updates only at the END of the iteration
            // (`alpha_update::update_all_alphas`, propagate_dag/mod.rs), so the
            // α folded into the loop-top segments is still current here. A stale
            // `iter`, a fired reference-bound refresh, or a malformed grads list
            // all reject the cache → fresh fold below (fail closed).
            let cache_hit = reusable_cache;
            let used_cache = cache_hit.is_some();
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
                    let pre_lowers = self.warmup_masked_pre_lowers(
                        ctx,
                        node_bounds,
                        runtime.graph(),
                        &relu_names,
                    )?;
                    let result = {
                        if deadline.is_some_and(|value| Instant::now() >= value) {
                            return if margin_request.is_armed() {
                                Some(WarmupGradientResult::margin_supplied_fallback(
                                    local_gradients,
                                    MarginGradientFallbackReason::DeadlineExpired,
                                ))
                            } else {
                                None
                            };
                        }
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
                    if result.relu_grads.len() != relu_names.len()
                        || result
                            .relu_grads
                            .iter()
                            .flatten()
                            .any(|value| !value.is_finite())
                        || !crate::sound_gpu_gate::gpu_interval_payload_is_publishable(
                            &result.lower_bounds,
                            &result.upper_bounds,
                            seed.num_specs,
                        )
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
            let resident_fallback = |reason| {
                Self::map_warmup_fold_gradients(
                    ctx,
                    runtime.graph(),
                    local_gradients,
                    relu_names,
                    local_rule_grads,
                )
                .map_or_else(
                    || {
                        WarmupGradientResult::margin_supplied_fallback(
                            local_gradients,
                            MarginGradientFallbackReason::MappingMismatch,
                        )
                    },
                    |mapped| WarmupGradientResult::margin_resident_fallback(mapped, reason),
                )
            };
            let deadline_expired = deadline.is_some_and(|value| Instant::now() >= value);
            if post_cache_deadline_refusal_required(
                margin_request.is_armed(),
                used_cache,
                deadline_expired,
            ) {
                if margin_request.is_armed() {
                    return Some(resident_fallback(
                        MarginGradientFallbackReason::DeadlineExpired,
                    ));
                }
                return None;
            }
            if matches!(margin_request, MarginGradientRequest::NoBinding) {
                return Some(resident_fallback(MarginGradientFallbackReason::NoBinding));
            }

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
            let (relu_grads_fold, margin_dispatch): (Vec<Vec<f32>>, MarginGradientDispatch) =
                if let Some(spec) = margin_objective {
                    // #root-alpha-margin-gradient (dark): steer this update with the
                    // current binding verification row, not the identity-seed sum of
                    // raw logits.  The already-folded segments contain this iterate's
                    // exact alpha state; one GPU-resident whole-fold adjoint computes
                    // the direct-C lower-objective gradient.  No CPU replay fallback
                    // is permitted here because it can overrun the root deadline.
                    //
                    // Soundness: this value only proposes alpha in [0,1]. The next
                    // loop iteration independently recomputes sound CROWN bounds, and
                    // the owning loop retains one complete state/bound-score pair.
                    if deadline.is_some_and(|value| Instant::now() >= value) {
                        // An attempted child objective must not fall through into the
                        // unbounded CPU AnalyticChain path after consuming its budget.
                        return Some(resident_fallback(
                            MarginGradientFallbackReason::DeadlineExpired,
                        ));
                    }
                    let Some(joint_deadline) = deadline else {
                        return Some(resident_fallback(
                            MarginGradientFallbackReason::ResidentUnavailable,
                        ));
                    };
                    if !gpu.provides_deadline_bounded_joint_alpha_gradient_resident() {
                        return Some(resident_fallback(
                            MarginGradientFallbackReason::ResidentUnavailable,
                        ));
                    }
                    let margin_grads = joint_alpha_grads_fold_gpu_with_deadline(
                        gpu,
                        segments,
                        spec,
                        1,
                        od,
                        &in_lo,
                        &in_hi,
                        pre_lowers,
                        relu_names.len(),
                        joint_deadline,
                    );
                    match margin_grads {
                        Ok(grads) => {
                            if probe {
                                eprintln!(
                                    "[warmup-grad] margin objective GPU SUCCESS width={}",
                                    spec.len()
                                );
                            }
                            (
                                grads,
                                MarginGradientDispatch::JointDispatched {
                                    source: MarginGradientJointSource::Resident,
                                },
                            )
                        }
                        Err(error) => {
                            if probe {
                                eprintln!(
                            "[warmup-grad] margin objective unavailable → bounded local fallback"
                        );
                            }
                            return Some(resident_fallback(deadline_joint_fold_fallback_reason(
                                error,
                            )));
                        }
                    }
                } else if multiobj_joint_alpha_enabled() {
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
                        if deadline.is_some_and(|value| Instant::now() >= value) {
                            return None;
                        }
                        let _deadline_scope =
                            crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(gpu, deadline);
                        let result =
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
                    );
                        if deadline.is_some_and(|value| Instant::now() >= value) {
                            return None;
                        }
                        result
                    } else {
                        None
                    };
                    let gradients = match gpu_joint {
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
                    };
                    (gradients, MarginGradientDispatch::NotRequested)
                } else if root_alpha_true_enabled() {
                    // #root-alpha-gpu (B): on a cache hit the replays run against the
                    // cached `segments`/`lower_bounds` — the same fold the loop-top
                    // bound came from, still α-current (see `WarmupGpuIterCache`).
                    let gradients = match self.true_root_warmup_gradients(
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
                    };
                    (gradients, MarginGradientDispatch::NotRequested)
                } else {
                    // #envelope-grad-gpu: the default configuration lands here,
                    // and `local_rule_grads` is the SIGN-DEFINITE field
                    // (`g <= 0` always ⇒ alpha walks monotonically into the 0
                    // clamp = the loosest envelope). When the envelope gate is
                    // armed, rescale it into the envelope rule on the host.
                    let envelope =
                        if crate::network::graph_alpha::backward::gradients::envelope_grad_enabled()
                        {
                            self.envelope_rescaled_warmup_gradients(
                                ctx,
                                node_bounds,
                                segments,
                                relu_names,
                                pre_lowers,
                                local_rule_grads,
                                lower_bounds,
                                &in_lo,
                                &in_hi,
                                deadline,
                            )
                        } else {
                            None
                        };
                    if probe {
                        eprintln!(
                            "[envelope-grad-gpu] {}",
                            if envelope.is_some() {
                                "RESCALED the resident local-rule gradients"
                            } else if crate::network::graph_alpha::backward::gradients::
                                envelope_grad_enabled()
                            {
                                "ARMED but unavailable -> local rule (fail-closed)"
                            } else {
                                "OFF -> local rule"
                            }
                        );
                    }
                    (
                        envelope.unwrap_or_else(|| local_rule_grads.to_vec()),
                        MarginGradientDispatch::NotRequested,
                    )
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
            if probe && !margin_request.is_armed() {
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
            let out = if margin_request.is_armed() {
                let Some(mapping_deadline) = deadline else {
                    return Some(resident_fallback(
                        MarginGradientFallbackReason::ResidentUnavailable,
                    ));
                };
                match Self::map_warmup_fold_gradients_with_deadline(
                    ctx,
                    runtime.graph(),
                    local_gradients,
                    relu_names,
                    &relu_grads_fold,
                    mapping_deadline,
                ) {
                    Ok(Some(out)) => out,
                    Ok(None) => {
                        return Some(resident_fallback(
                            MarginGradientFallbackReason::MappingMismatch,
                        ));
                    }
                    Err(error) => {
                        return Some(resident_fallback(deadline_joint_fold_fallback_reason(
                            error,
                        )));
                    }
                }
            } else {
                Self::map_warmup_fold_gradients(
                    ctx,
                    runtime.graph(),
                    local_gradients,
                    relu_names,
                    &relu_grads_fold,
                )?
            };
            debug!(
                relus = relu_names.len(),
                "DAG α-CROWN: GPU-resident resnet warmup gradients (step 3b-ii)"
            );
            if margin_request.is_armed() && deadline.is_some_and(|value| Instant::now() >= value) {
                return Some(resident_fallback(
                    MarginGradientFallbackReason::DeadlineExpired,
                ));
            }
            Some(if margin_dispatch.joint_dispatched() {
                WarmupGradientResult::margin_joint_resident(out)
            } else {
                WarmupGradientResult::legacy(out)
            })
        })();

        match attempt {
            Some(result) => Some(result),
            None if margin_request.is_armed() => {
                Some(WarmupGradientResult::margin_supplied_fallback(
                    local_gradients,
                    if matches!(margin_request, MarginGradientRequest::NoBinding) {
                        MarginGradientFallbackReason::NoBinding
                    } else {
                        MarginGradientFallbackReason::ResidentUnavailable
                    },
                ))
            }
            None => None,
        }
    }

    /// #alpha-steering-proposal: the PROPOSAL twin of the narrow-API CUDA
    /// branch in [`Self::try_gpu_resnet_warmup_gradients`] — one bounded,
    /// `num_specs=1` binding-row joint-α adjoint on the dedicated α-steering
    /// channel, entered only when NO verdict-authority resident backend
    /// exists (Metal).
    ///
    /// AUTHORITY HYGIENE. The engine on `ctx.alpha_steering` has no verdict
    /// authority and never acquires any here:
    /// - The ONLY method dispatched is
    ///   [`joint_alpha_grads_fold_gpu_with_deadline`] →
    ///   `crown_joint_alpha_gradient_resident_with_deadline`, whose return
    ///   type is `Vec<Vec<f32>>` gradients — no bound value exists on this
    ///   path to consume or discard (type-level; pinned by
    ///   `proposal_channel_never_invokes_bound_producing_methods`).
    /// - The output enters exclusively the `MarginGradientRequest` gradient
    ///   slot; the loop's own certified CPU fold evaluates every α iterate it
    ///   produces (`dag_alpha_backward_pass_with_engine`,
    ///   propagate_dag/mod.rs), and element-wise best-state retention
    ///   (`update_elementwise_best_bounds`) rejects regressions — a wrong
    ///   direction can only waste iterations, never a verdict.
    /// - Gradient screening is the SAME moat as the resident lane: NaN/Inf,
    ///   length, and mapping mismatches are typed refusals inside
    ///   [`joint_alpha_grads_fold_gpu_with_deadline`]; the deadline is polled
    ///   cooperatively and expiry refuses without a late publication.
    ///
    /// Returns `None` (channel absent / no deadline / extraction refusal) so
    /// the caller's outer moat preserves today's bounded local fallback
    /// byte-identically; typed refusals return the explicit
    /// `margin_supplied_fallback` exactly like the CUDA narrow branch.
    fn try_wgpu_proposal_joint_gradients(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &DagAlphaRuntimeState,
        local_gradients: &[Array1<f32>],
        objective: &[f32],
    ) -> Option<WarmupGradientResult> {
        let gpu = ctx.alpha_steering?.as_gpu_crown_backward()?;
        // The proposal API is deadline-bounded ONLY: without a deadline there
        // is no bounded contract to dispatch under (same rule as the resident
        // joint lane).
        let deadline = ctx.config.deadline?;
        let output_node = if self.output_node.is_empty() {
            ctx.exec_order.last()?.clone()
        } else {
            self.output_node.clone()
        };
        let (segments, relu_names, _frontier_abs, _node_abs) =
            self.warmup_segments(ctx, node_bounds, runtime, &output_node)?;
        let pre_lowers =
            self.warmup_masked_pre_lowers(ctx, node_bounds, runtime.graph(), &relu_names)?;
        let in_lo: Vec<f32> = ctx.input.lower().iter().copied().collect();
        let in_hi: Vec<f32> = ctx.input.upper().iter().copied().collect();
        let gradients = match joint_alpha_grads_fold_gpu_with_deadline(
            gpu,
            &segments,
            objective,
            1,
            ctx.output_dim,
            &in_lo,
            &in_hi,
            &pre_lowers,
            relu_names.len(),
            deadline,
        ) {
            Ok(gradients) => gradients,
            Err(error) => {
                return Some(WarmupGradientResult::margin_supplied_fallback(
                    local_gradients,
                    deadline_joint_fold_fallback_reason(error),
                ));
            }
        };
        let mapped = match Self::map_warmup_fold_gradients_with_deadline(
            ctx,
            runtime.graph(),
            local_gradients,
            &relu_names,
            &gradients,
            deadline,
        ) {
            Ok(Some(mapped)) => mapped,
            Ok(None) => {
                return Some(WarmupGradientResult::margin_supplied_fallback(
                    local_gradients,
                    MarginGradientFallbackReason::MappingMismatch,
                ));
            }
            Err(error) => {
                return Some(WarmupGradientResult::margin_supplied_fallback(
                    local_gradients,
                    deadline_joint_fold_fallback_reason(error),
                ));
            }
        };
        if Instant::now() >= deadline {
            return Some(WarmupGradientResult::margin_supplied_fallback(
                local_gradients,
                MarginGradientFallbackReason::DeadlineExpired,
            ));
        }
        crate::alpha_gradient_steering::note_proposal_dispatch();
        Some(WarmupGradientResult::margin_joint_proposal(mapped))
    }

    /// #binding-row-replay: the PRODUCTION tier of the CPU binding-row replay
    /// (`binding_row_true_alpha_grads`, FD-proven cosine 0.999998 per-neuron /
    /// 1.000000 channel-shared vs central differences of the certified fold) —
    /// entered only on the armed margin lane of a host WITHOUT a
    /// verdict-authority resident backend, after the proposal channel was
    /// absent or typed-refused (I10: no new env flags; unreachable when the
    /// child gates are disarmed).
    ///
    /// One `dag_alpha_backward_pass_with_intermediates` — the SAME pass the
    /// child-disabled AnalyticChain lane runs every gradient iteration on
    /// these hosts — captures the dense A-matrices; the replay borrows the
    /// intermediate (no clone of `a_at_relu`, no second pass) and costs
    /// ~50ms/binding row on top (quiet p95 50.3ms, arbitration doc §6). The
    /// pass runs under the loop deadline, so an armed iteration stays bounded:
    /// a deadline-truncated capture is a typed missing-`a_at_relu` refusal
    /// here, never a late publication.
    ///
    /// THE SEED-ROW TRAP (arbitration doc §replay caveat): `binding_row`
    /// indexes the SEED row space of the capturing fold, and the objective is
    /// an OUTPUT-space row. The two reconcile only when the objective reads a
    /// single output row with a positive weight (`lb(s·e_r) = s·lb(e_r)` for
    /// `s > 0` — a general combination is NOT a combination of per-row
    /// gradients, each row selects its own relaxation branches), mapped
    /// through the published #margin-subset-alpha scope when the capture was
    /// subset-seeded — keyed on the MEASURED captured seed-row count, never a
    /// config flag. Anything else is a typed refusal.
    ///
    /// SOUNDNESS: identical moat to the proposal channel — the output enters
    /// exclusively the margin-gradient slot, every α iterate is re-evaluated
    /// by the certified fold, and element-wise best-state retention rejects
    /// regressions. Additionally the loop clamps replay-sourced updates to
    /// max|Δα| ≤ 0.05 (consult #6 v1 trust region, cheap half).
    ///
    /// Returns `None` on any typed refusal (reason logged at the
    /// #root-alpha-margin-gradient marker) so the caller falls through to the
    /// next tier with the prior tier's reason preserved.
    fn try_binding_row_replay_gradients(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &DagAlphaRuntimeState,
        local_gradients: &[Array1<f32>],
        objective: &[f32],
    ) -> Option<WarmupGradientResult> {
        fn refuse(reason: &str) -> Option<WarmupGradientResult> {
            info!(
                "DAG α-CROWN #root-alpha-margin-gradient: \
                 dispatch=binding_row_replay refused reason={reason}"
            );
            None
        }

        // The replay differentiates ReLU α of the certified fold only;
        // other optimizable-alpha families are outside its envelope.
        // (Margin-gradient eligibility already requires a ReLU-only
        // optimizer state — this is the defensive twin of that gate.)
        if ctx.has_bilinear || ctx.has_mul_binary {
            return refuse("non_relu_alpha_family");
        }
        // Same bounded-contract rule as the proposal channel and the resident
        // joint lane: no deadline ⇒ no bounded dispatch ⇒ refuse.
        let Some(deadline) = ctx.config.deadline else {
            return refuse("no_deadline");
        };
        if Instant::now() >= deadline {
            return refuse("deadline_expired");
        }
        let mut seed_row_probe = false;
        let (output_row, scale) = match single_positive_objective_row(objective) {
            Some(pair) => pair,
            None => match replay_row_index_enabled().then(binding_seed_row).flatten() {
                // #replay-row-index PROBE: the margin objective is a spec-row
                // combination in LOGIT space, but the captured seed rows ARE
                // the spec rows, so the ascent's row index is the seed row.
                Some(row) => {
                    seed_row_probe = true;
                    (row, 1.0f32)
                }
                None => return refuse("objective_not_single_positive_row"),
            },
        };

        // The AnalyticChain intermediates capture, deadline-bounded.
        let mut scratch = zeros_like_gradients(local_gradients);
        let mut scratch_upper = zeros_like_gradients(local_gradients);
        let intermediate = match self.dag_alpha_backward_pass_with_intermediates(
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
            None,
            None,
            Some(deadline),
        ) {
            Ok((_bounds, intermediate)) => intermediate,
            Err(error) => {
                debug!("#binding-row-replay intermediates capture failed: {error}");
                return refuse("intermediates_capture_failed");
            }
        };

        // Objective output row → capture seed row, keyed on the MEASURED
        // captured seed-row count (the documented margin-subset trap).
        let seed_rows = intermediate.final_bounds.lower_a().nrows();
        let binding_row = if seed_row_probe {
            // #replay-row-index PROBE: `output_row` is ALREADY a seed-row index
            // (it came from the ascent's row ordering), so the logit→seed
            // remapping below must not run. Range-check and use it directly.
            info!(
                "DAG α-CROWN #root-alpha-margin-gradient: dispatch=binding_row_replay \
                 probe seed_rows={seed_rows} output_dim={} row={output_row}",
                ctx.output_dim
            );
            if output_row >= seed_rows {
                return refuse("probe_row_out_of_seed_range");
            }
            output_row
        } else if seed_rows == ctx.output_dim {
            output_row
        } else {
            let Some(indices) = crate::output_margin_seed::margin_subset_indices(ctx.output_dim)
            else {
                return refuse("seed_row_space_unmapped");
            };
            if indices.len() != seed_rows {
                return refuse("seed_row_space_unmapped");
            }
            match indices
                .iter()
                .position(|&published| published == output_row)
            {
                Some(compact) => compact,
                None => return refuse("binding_row_outside_margin_subset"),
            }
        };

        let replay = match self.binding_row_true_alpha_grads(
            ctx.input,
            runtime.graph(),
            &intermediate,
            binding_row,
        ) {
            Ok(replay) => replay,
            Err(error) => {
                debug!("#binding-row-replay typed refusal: {error}");
                return refuse("replay_typed_refusal");
            }
        };

        // Assemble in runtime ReLU order at ALPHA width; ReLUs outside the
        // captured fold keep the supplied local gradient (the resident
        // mapper's convention). The replay already emits alpha-width vectors
        // (channel-summed for channel-shared α), so any width mismatch is an
        // irreconcilable geometry, not a reducible one.
        let mut out = local_gradients.to_vec();
        for ((name, _), slot) in ctx.relu_nodes.iter().zip(out.iter_mut()) {
            let Some(grad) = replay.grads.get(name.as_str()) else {
                continue;
            };
            if grad.len() != slot.len() {
                return refuse("irreconcilable_alpha_geometry");
            }
            // lb(s·e_r) = s·lb(e_r) for s > 0: the exact chain rule through
            // the objective's positive scale. `s == 1.0` keeps the replay
            // output bit-identical (steering-grade exactness oracle).
            *slot = if scale == 1.0 {
                grad.clone()
            } else {
                grad.mapv(|value| value * scale)
            };
        }
        if out.iter().flatten().any(|value| !value.is_finite()) {
            return refuse("nonfinite_gradient");
        }
        if Instant::now() >= deadline {
            return refuse("deadline_expired");
        }
        crate::alpha_gradient_steering::note_replay_dispatch();
        Some(WarmupGradientResult::margin_binding_row_replay(out))
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
        let deadline = ctx.config.deadline;
        if !crate::sound_gpu_gate::gpu_crown_backend_honors_deadline(gpu, deadline)
            || deadline.is_some_and(|value| Instant::now() >= value)
        {
            return None;
        }
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
        if deadline.is_some_and(|value| Instant::now() >= value) {
            return None;
        }
        let result = {
            let _deadline_scope =
                crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(gpu, deadline);
            gpu.crown_backward_gpu_resnet_sound(
                &segments,
                &seed,
                &in_lo,
                &in_hi,
                &frontier_abs,
                &node_abs,
            )
            .ok()?
        };
        if deadline.is_some_and(|value| Instant::now() >= value) {
            return None;
        }
        if !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&result, od) {
            return None;
        }
        let lower =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[od]), result.lower_bounds).ok()?;
        let upper =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[od]), result.upper_bounds).ok()?;
        BoundedTensor::new(lower, upper).ok()
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
        let deadline = ctx.config.deadline;
        if !crate::sound_gpu_gate::gpu_crown_backend_honors_deadline(gpu, deadline)
            || deadline.is_some_and(|value| Instant::now() >= value)
        {
            return None;
        }
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
        if !crate::sound_gpu_gate::gpu_interval_payload_is_publishable(
            &result.lower_bounds,
            &result.upper_bounds,
            od,
        ) || result
            .relu_grads
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
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
        let bounds = BoundedTensor::new(lower, upper).ok()?;
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

    /// #envelope-grad-gpu: rescale the resident LOCAL-RULE alpha gradients into
    /// the ENVELOPE rule ON THE HOST — no kernel change, no second fold, and no
    /// change to any buffer the device reads.
    ///
    /// The device computes `g[i] = pre_lower[i] * S[i]` with
    /// `S[i] = sum_j max(a_lower[j,i], 0)` (`CROWN_ALPHA_GRADIENT_SHADER`). The
    /// CPU envelope rule is `factor[i] * S[i]` with
    /// `factor[i] = h_hat_i(x*).clamp(l_i, u_i)` (`backward/gradients.rs`, the
    /// `let factor = match hhat` block). `S` is the expensive part and is
    /// IDENTICAL in both, so recovering it as `g / l` and re-multiplying by the
    /// envelope factor reproduces the CPU rule exactly — including on the
    /// cache-hit path, where no fold runs at all.
    ///
    /// WHY NOT REWRITE THE `pre_lowers` BUFFER INSTEAD. Two load-bearing
    /// reasons: (1) that buffer is ALSO the stable-neuron MASK for the
    /// joint-adjoint lanes and for `true_root_warmup_gradients`, all keyed on
    /// `pl[i] == 0.0` — an envelope factor landing on exactly `0.0` at an
    /// UNSTABLE neuron would silently zero a true gradient; (2) at the bound
    /// fold the buffer is an INPUT to the very fold whose `lower_bounds` name
    /// the binding row, so a factor built there could only use a STALE row.
    /// Here `lower_bounds` is this iterate's own.
    ///
    /// `None` leaves the caller on the local rule. Fail-closed is cheap: this
    /// is a CORRECTNESS path, not a soundness one — gradients only steer
    /// `alpha in [0,1]`, every value of which is a certified-sound relaxation.
    ///
    /// # STATUS: WIRED AND FIRING, BUT *NOT* YET EQUIVALENT TO THE CPU RULE
    ///
    /// Measured 2026-08-12 on cifar100 idx_7641, interior alphas per ascent
    /// iteration, all three on the same binary:
    ///
    /// ```text
    /// CPU  + envelope   125 126 122 125  26  21   <- the target behaviour
    /// GPU  + local       63  63  63  63   0   0
    /// GPU  + THIS RULE   51  51  51  51   0   1   <- fires 11x, still collapses
    /// ```
    ///
    /// So the rescale demonstrably CHANGES the field (63 -> 51 at iter 0) but
    /// does not reproduce the CPU envelope rule, and alpha still reaches the 0
    /// clamp. The algebra `factor*S == (g/l)*factor` is exact, so the divergence
    /// is in one of the INPUTS, not the rescale: prime suspects are (a) `x*` —
    /// this lane names it from `binding_row_argmin_corner` over the fold's own
    /// `lower_bounds`, whereas the CPU lane derives it from
    /// `intermediate.final_bounds`, and the two need not agree; (b) which spec
    /// rows the resident fold actually summed into `S`. Diagnose by dumping both
    /// lanes' `x*` for the same iterate and diffing them BEFORE trusting any
    /// verdict measured through this path.
    #[allow(clippy::too_many_arguments)]
    fn envelope_rescaled_warmup_gradients(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        segments: &[GpuResnetSegment],
        relu_names: &[String],
        pre_lowers: &[Vec<f32>],
        local_rule_grads: &[Vec<f32>],
        lower_bounds: &[f32],
        in_lo: &[f32],
        in_hi: &[f32],
        deadline: Option<Instant>,
    ) -> Option<Vec<Vec<f32>>> {
        if local_rule_grads.len() != relu_names.len() || pre_lowers.len() != relu_names.len() {
            return None;
        }
        // Binding row = the smallest FINITE concretized lower bound, the same
        // criterion `envelope_binding_points` applies to
        // `intermediate.final_bounds`. The warmup seed is the identity over
        // `od` (`identity_warmup_seed`), so fold row r IS output row r.
        // `min_by` returns the FIRST minimum, matching the CPU's strict-`<`
        // first-wins tie rule.
        let (binding_row, _) = lower_bounds
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .min_by(|(_, a), (_, b)| a.total_cmp(b))?;
        let mut spec_row = vec![0.0f32; lower_bounds.len()];
        *spec_row.get_mut(binding_row)? = 1.0;

        let x_star = crate::beta_crown::engine::graph::propagation::batched::wide_alpha_true::
            binding_row_argmin_corner(segments, &spec_row, in_lo, in_hi, deadline)?;
        if crate::network::graph_alpha::backward::gradients::envelope_xstar_probe_enabled() {
            eprintln!(
                "[xstar-gpu] {}",
                crate::network::graph_alpha::backward::gradients::summarize_x_star(
                    binding_row,
                    &x_star
                )
            );
        }
        let points = self.envelope_points_at(ctx.input, relu_names, &x_star, ctx.engine)?;

        let mut out: Vec<Vec<f32>> = Vec::with_capacity(relu_names.len());
        for (k, name) in relu_names.iter().enumerate() {
            let pl = &pre_lowers[k];
            let g = &local_rule_grads[k];
            if g.len() != pl.len() {
                return None;
            }
            // Per-LAYER fallback, mirroring the CPU rule's
            // `.and_then(|m| m.get(relu_name)).filter(|h| h.len() == n)`. A
            // whole-map refusal here would make the two lanes different rules.
            let hhat = points.get(name).filter(|h| h.len() == pl.len());
            let Some(hhat) = hhat else {
                out.push(g.clone());
                continue;
            };
            let upper = self
                .nodes
                .get(name)
                .and_then(|node| node.inputs.first())
                .and_then(|input_name| {
                    if input_name == NETWORK_INPUT {
                        Some(ctx.input)
                    } else {
                        node_bounds.get(input_name)
                    }
                })
                .and_then(|pre| pre.upper().as_slice().map(<[f32]>::to_vec));
            let Some(upper) = upper.filter(|u| u.len() == pl.len()) else {
                out.push(g.clone());
                continue;
            };
            // #envelope-grad-gpu DIAGNOSTIC: the rescale assumes the consumed
            // `g` is still the RAW kernel product `l * S` with `S >= 0`. If any
            // post-processing (a negation, a per-channel reduction, a clamp)
            // sits between the kernel and here, `g / l` is not `S` and the
            // rescale is differently-wrong rather than right. `g <= 0` for every
            // unstable neuron is the cheap falsifier for that assumption.
            if k == 0
                && crate::network::graph_alpha::backward::gradients::envelope_rescale_probe_enabled(
                )
            {
                let (mut unstable, mut positive_g) = (0usize, 0usize);
                let mut sample = String::new();
                for i in 0..pl.len() {
                    if pl[i] == 0.0 || !pl[i].is_finite() {
                        continue;
                    }
                    unstable += 1;
                    if g[i] > 0.0 {
                        positive_g += 1;
                    }
                    if sample.len() < 220 {
                        let s = g[i] / pl[i];
                        let f = if hhat[i].is_finite() {
                            hhat[i].clamp(pl[i], upper[i])
                        } else {
                            pl[i]
                        };
                        sample.push_str(&format!(
                            "(l={:.3e} u={:.3e} g={:.3e} S={:.3e} h={:.3e} f={:.3e} out={:.3e}) ",
                            pl[i],
                            upper[i],
                            g[i],
                            s,
                            hhat[i],
                            f,
                            s * f
                        ));
                    }
                }
                // S-STATISTICS: the last remaining suspect for the divergence.
                // `S >= 0` in both lanes so it cannot flip a gradient's SIGN —
                // it can only ZERO one, which leaves that alpha pinned at its
                // {0,1} init and therefore never interior. Count the zeros.
                let (mut s_zero, mut s_sum) = (0usize, 0.0f64);
                for i in 0..pl.len() {
                    if pl[i] == 0.0 || !pl[i].is_finite() {
                        continue;
                    }
                    let s = g[i] / pl[i];
                    if s == 0.0 {
                        s_zero += 1;
                    }
                    s_sum += f64::from(s);
                }
                eprintln!(
                    "[envelope-rescale] relu0 unstable={unstable} g_positive={positive_g} \
                     S_zero={s_zero} S_mean={:.4e} \
                     (g>0 at an unstable neuron falsifies g == l*S with S>=0)\n  {sample}",
                    s_sum / f64::from(u32::try_from(unstable.max(1)).unwrap_or(1))
                );
            }
            let mut row = Vec::with_capacity(pl.len());
            for i in 0..pl.len() {
                let l = pl[i];
                let u = upper[i];
                // `pl[i] == 0.0` is the STABLE-neuron mask; its gradient is
                // already 0 and dividing by it is the one arithmetic trap here.
                if l == 0.0 || !l.is_finite() || !u.is_finite() || !g[i].is_finite() {
                    row.push(g[i]);
                    continue;
                }
                // Divide FIRST. `factor / l` would overflow for a denormal `l`;
                // `g / l` recovers `S` at the magnitude of the real quantity.
                let s = g[i] / l;
                let factor = if hhat[i].is_finite() {
                    hhat[i].clamp(l, u)
                } else {
                    l
                };
                let scaled = s * factor;
                row.push(if scaled.is_finite() { scaled } else { g[i] });
            }
            out.push(row);
        }
        Some(out)
    }
}

/// #binding-row-replay: the only objective shape a SINGLE seed-row replay
/// represents exactly — one nonzero entry with a positive finite weight
/// (`d lb(s·e_r)/dα = s · d lb(e_r)/dα` for `s > 0`; a multi-row combination
/// selects its own relaxation branches and is NOT a combination of per-row
/// gradients). Returns `(output_row, scale)` or `None` (typed refusal at the
/// caller).
fn single_positive_objective_row(objective: &[f32]) -> Option<(usize, f32)> {
    let mut nonzero: Option<(usize, f32)> = None;
    for (index, &value) in objective.iter().enumerate() {
        if value != 0.0 {
            if nonzero.is_some() {
                return None;
            }
            nonzero = Some((index, value));
        }
    }
    let (output_row, scale) = nonzero?;
    (scale.is_finite() && scale > 0.0).then_some((output_row, scale))
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
    use crate::bounds::{AlphaCrownConfig, AlphaSpecAscent, AlphaSpecEarlyExit};
    use crate::network::graph_alpha::resnet_skeleton::build_resnet_segment_skeleton;
    use crate::network::graph_alpha::resnet_skeleton::test_support::{
        assert_extraction_bits_eq, box_input, conv_resnet_fixture, mk_alpha, relu, static_arc_ptrs,
        CONV_FIXTURE_RELUS,
    };
    use ny_core::{GpuCrownGradResult, GpuCrownLayer, GpuCrownResult, NaiveCpuGemmEngine, NyError};
    use ny_test_utils::env::with_env_edits;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Duration;

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
        ctx_with_steering(fix, engine, None)
    }

    fn ctx_with_steering<'a>(
        fix: &'a WarmupFixture,
        engine: Option<&'a dyn ny_core::GemmEngine>,
        alpha_steering: Option<&'a dyn ny_core::GemmEngine>,
    ) -> DagAlphaLoopContext<'a> {
        DagAlphaLoopContext {
            input: &fix.input,
            exec_order: &fix.exec_order,
            output_dim: fix.output_dim,
            input_dim: fix.input_dim,
            config: &fix.config,
            engine,
            alpha_steering,
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

    fn sentinel_local_grads(fix: &WarmupFixture) -> Vec<Array1<f32>> {
        fix.relu_nodes
            .iter()
            .enumerate()
            .map(|(relu_index, (_, len))| {
                Array1::from_iter((0..*len).map(|neuron_index| {
                    1000.0 + relu_index as f32 * 10.0 + neuron_index as f32 * 0.01
                }))
            })
            .collect()
    }

    fn grads_bits(grads: &[Array1<f32>]) -> Vec<Vec<u32>> {
        grads
            .iter()
            .map(|g| g.iter().map(|v| v.to_bits()).collect())
            .collect()
    }

    fn assert_local_margin_fallback(
        result: Option<WarmupGradientResult>,
        expected_gradients: &[Array1<f32>],
        reason: MarginGradientFallbackReason,
        source: MarginGradientFallbackSource,
    ) {
        let result = result.expect("an armed request must resolve inside the bounded lane");
        assert_eq!(
            result.margin_dispatch,
            MarginGradientDispatch::LocalFallback { reason, source }
        );
        assert_eq!(
            grads_bits(&result.gradients),
            grads_bits(expected_gradients),
            "the fallback gradients must match the source named by telemetry exactly"
        );
    }

    fn tensor_bits(bounds: &BoundedTensor) -> (Vec<usize>, Vec<u32>, Vec<u32>) {
        (
            bounds.shape().to_vec(),
            bounds.lower().iter().map(|v| v.to_bits()).collect(),
            bounds.upper().iter().map(|v| v.to_bits()).collect(),
        )
    }

    fn scripted_relu_widths(segments: &[GpuResnetSegment]) -> Vec<usize> {
        fn append_layer_widths(layers: &[GpuCrownLayer], widths: &mut Vec<usize>) {
            for layer in layers {
                match layer {
                    GpuCrownLayer::Activation { num_neurons, .. }
                    | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => {
                        widths.push(*num_neurons);
                    }
                    _ => {}
                }
            }
        }

        let mut widths = Vec::new();
        for segment in segments {
            match segment {
                GpuResnetSegment::Chain(layers) | GpuResnetSegment::Residual(layers) => {
                    append_layer_widths(layers, &mut widths);
                }
                GpuResnetSegment::ResidualProj(main, projection) => {
                    append_layer_widths(main, &mut widths);
                    append_layer_widths(projection, &mut widths);
                }
            }
        }
        widths
    }

    /// Deterministic CPU stand-in for the sound GPU resnet backward. The
    /// returned bounds/gradients are pure functions of the call inputs
    /// (masked pre-lowers, segment count, seed size, input box), so
    /// cache-vs-fresh comparisons are exact bitwise oracles, and the
    /// invocation counter pins "one kernel fold per iteration".
    #[derive(Clone, Copy, Debug, Default)]
    enum ScriptedJointBehavior {
        #[default]
        Valid,
        Error,
        NonFinite,
        WrongReluCount,
        WrongNeuronCount,
        ScriptedTailPastDeadline,
        /// Adversarially WRONG but finite direction: the exact negation of
        /// `Valid` (#alpha-steering-proposal best-state retention proof).
        SignFlipped,
        /// VERIFIER variant of `NonFinite`: poison the DEEPEST layer instead
        /// of the first (last ReLU, last neuron) and with `-inf` instead of
        /// NaN — proves the finiteness moat scans the whole gradient field,
        /// not just the head, and rejects every non-finite class.
        NegInfDeepLayer,
    }

    #[derive(Default)]
    struct ScriptedResnetGradEngine {
        grad_calls: AtomicUsize,
        joint_calls: AtomicUsize,
        joint_seeds: Mutex<Vec<Vec<f32>>>,
        joint_behavior: ScriptedJointBehavior,
        joint_widths: Mutex<Option<Vec<usize>>>,
        joint_work_units: Mutex<Vec<usize>>,
        cooperative_deadline: bool,
        cooperative_joint_deadline: bool,
        deadline_writes: Mutex<Vec<Option<Instant>>>,
        /// Model a PROPOSAL-grade engine (the wgpu α-steering wrapper): the
        /// joint adjoint is live while `provides_sound_gpu_crown()` is false.
        no_sound_authority: bool,
    }

    impl ScriptedResnetGradEngine {
        fn cooperative() -> Self {
            Self {
                cooperative_deadline: true,
                cooperative_joint_deadline: true,
                ..Self::default()
            }
        }

        fn joint_deadline_only() -> Self {
            Self {
                cooperative_joint_deadline: true,
                ..Self::default()
            }
        }

        fn joint_deadline_only_with_behavior(joint_behavior: ScriptedJointBehavior) -> Self {
            Self {
                joint_behavior,
                cooperative_joint_deadline: true,
                ..Self::default()
            }
        }

        fn cooperative_with_joint_behavior(joint_behavior: ScriptedJointBehavior) -> Self {
            Self {
                joint_behavior,
                cooperative_deadline: true,
                cooperative_joint_deadline: true,
                ..Self::default()
            }
        }

        /// The α-steering channel shape: deadline-bounded joint adjoint live,
        /// NO verdict authority (mirrors `ny_gpu::GradientSteeringDevice`,
        /// whose backward device keeps `provides_sound_gpu_crown() == false`).
        fn proposal_only_with_behavior(joint_behavior: ScriptedJointBehavior) -> Self {
            Self {
                joint_behavior,
                cooperative_joint_deadline: true,
                no_sound_authority: true,
                ..Self::default()
            }
        }

        fn set_joint_widths_from_cache(&self, cache: &WarmupGpuIterCache) {
            self.set_joint_widths(cache.pre_lowers.iter().map(Vec::len).collect());
        }

        fn set_joint_widths(&self, widths: Vec<usize>) {
            *self.joint_widths.lock().expect("joint_widths mutex") = Some(widths);
        }

        fn calls(&self) -> usize {
            self.grad_calls.load(Ordering::SeqCst)
        }

        fn joint_calls(&self) -> usize {
            self.joint_calls.load(Ordering::SeqCst)
        }

        fn joint_seeds(&self) -> Vec<Vec<f32>> {
            self.joint_seeds.lock().expect("joint_seeds mutex").clone()
        }

        fn deadline_writes(&self) -> Vec<Option<Instant>> {
            self.deadline_writes
                .lock()
                .expect("deadline_writes mutex")
                .clone()
        }

        fn joint_work_units(&self) -> Vec<usize> {
            self.joint_work_units
                .lock()
                .expect("joint_work_units mutex")
                .clone()
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
            !self.no_sound_authority
        }

        fn honors_crown_backward_deadline(&self) -> bool {
            self.cooperative_deadline
        }

        fn provides_deadline_bounded_joint_alpha_gradient_resident(&self) -> bool {
            self.cooperative_joint_deadline
        }

        fn set_crown_backward_deadline(&self, deadline: Option<Instant>) {
            self.deadline_writes
                .lock()
                .expect("deadline_writes mutex")
                .push(deadline);
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

        fn crown_joint_alpha_gradient_resident(
            &self,
            segments: &[GpuResnetSegment],
            seed_lower_a: &[f32],
            num_specs: usize,
            output_dim: usize,
            input_lower: &[f32],
            input_upper: &[f32],
        ) -> Result<Vec<Vec<f32>>> {
            self.joint_calls.fetch_add(1, Ordering::SeqCst);
            self.joint_seeds
                .lock()
                .expect("joint_seeds mutex")
                .push(seed_lower_a.to_vec());
            if matches!(self.joint_behavior, ScriptedJointBehavior::Error) {
                return Err(NyError::UnsupportedOp(
                    "scripted engine: resident joint refusal".into(),
                ));
            }
            assert_eq!(seed_lower_a.len(), num_specs * output_dim);
            assert_eq!(input_lower.len(), input_upper.len());
            let widths = self
                .joint_widths
                .lock()
                .expect("joint_widths mutex")
                .clone()
                .unwrap_or_else(|| scripted_relu_widths(segments));
            let mut gradients: Vec<Vec<f32>> =
                widths.into_iter().map(|width| vec![0.0; width]).collect();
            // Make the scripted backend's routed objective observable even
            // when this tiny fixture happens to sit at a zero-gradient alpha.
            let seed_fingerprint: f32 = seed_lower_a
                .iter()
                .enumerate()
                .map(|(index, value)| (index + 1) as f32 * value)
                .sum();
            for (relu_index, relu) in gradients.iter_mut().enumerate() {
                for (neuron_index, value) in relu.iter_mut().enumerate() {
                    *value += seed_fingerprint
                        * (relu_index + 1) as f32
                        * (neuron_index + 1) as f32
                        * 1e-4;
                }
            }
            match self.joint_behavior {
                ScriptedJointBehavior::Valid | ScriptedJointBehavior::Error => {}
                ScriptedJointBehavior::NonFinite => {
                    if let Some(value) = gradients.first_mut().and_then(|relu| relu.first_mut()) {
                        *value = f32::NAN;
                    }
                }
                ScriptedJointBehavior::WrongReluCount => {
                    gradients.pop();
                }
                ScriptedJointBehavior::WrongNeuronCount => {
                    if let Some(relu) = gradients.first_mut() {
                        relu.push(0.0);
                    }
                }
                ScriptedJointBehavior::ScriptedTailPastDeadline => {}
                ScriptedJointBehavior::SignFlipped => {
                    for relu in &mut gradients {
                        for value in relu {
                            *value = -*value;
                        }
                    }
                }
                ScriptedJointBehavior::NegInfDeepLayer => {
                    if let Some(value) = gradients.last_mut().and_then(|relu| relu.last_mut()) {
                        *value = f32::NEG_INFINITY;
                    }
                }
            }
            Ok(gradients)
        }

        fn crown_joint_alpha_gradient_resident_with_deadline(
            &self,
            segments: &[GpuResnetSegment],
            seed_lower_a: &[f32],
            num_specs: usize,
            output_dim: usize,
            input_lower: &[f32],
            input_upper: &[f32],
            deadline: Instant,
        ) -> Result<Vec<Vec<f32>>> {
            if !self.cooperative_joint_deadline {
                return Err(NyError::UnsupportedOp(
                    "scripted engine: no joint deadline capability".into(),
                ));
            }
            if matches!(
                self.joint_behavior,
                ScriptedJointBehavior::ScriptedTailPastDeadline
            ) {
                self.joint_calls.fetch_add(1, Ordering::SeqCst);
                self.joint_seeds
                    .lock()
                    .expect("joint_seeds mutex")
                    .push(seed_lower_a.to_vec());
                for unit in 0..4 {
                    if Instant::now() >= deadline {
                        return Err(NyError::DeadlineExceeded(
                            "scripted joint tail cancelled".into(),
                        ));
                    }
                    self.joint_work_units
                        .lock()
                        .expect("joint_work_units mutex")
                        .push(unit);
                    if unit == 0 {
                        std::thread::sleep(Duration::from_millis(30));
                    }
                }
                return Err(NyError::InternalError(
                    "scripted deadline was too long to exercise cancellation".into(),
                ));
            }
            if Instant::now() >= deadline {
                return Err(NyError::DeadlineExceeded(
                    "scripted joint deadline expired before work".into(),
                ));
            }
            let result = self.crown_joint_alpha_gradient_resident(
                segments,
                seed_lower_a,
                num_specs,
                output_dim,
                input_lower,
                input_upper,
            )?;
            if Instant::now() >= deadline {
                return Err(NyError::DeadlineExceeded(
                    "scripted joint deadline expired after work".into(),
                ));
            }
            Ok(result)
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

    #[test]
    fn post_cache_deadline_guard_preserves_disabled_cache_hit_exactly() {
        assert!(
            !post_cache_deadline_refusal_required(false, true, true),
            "the pre-feature disabled cache-hit path must ignore the new post-cache guard"
        );
        assert!(
            post_cache_deadline_refusal_required(false, false, true),
            "the historical disabled fresh-fold refusal must remain"
        );
        assert!(
            post_cache_deadline_refusal_required(true, true, true),
            "an armed margin request must refuse a late cached result"
        );
        assert!(
            !post_cache_deadline_refusal_required(true, false, false),
            "a live deadline must not refuse either path"
        );
    }

    #[test]
    fn margin_fallback_sources_have_distinct_sealed_telemetry_labels() {
        assert_eq!(
            MarginGradientFallbackSource::ResidentLocalRule.as_str(),
            "resident_local_rule"
        );
        assert_eq!(
            MarginGradientFallbackSource::SuppliedLocal.as_str(),
            "supplied_local"
        );
    }

    #[test]
    fn warmup_gpu_deadline_admission_and_exact_scope() {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_BETA_GPU_PROBE");

            let mut expired_fix = warmup_fixture();
            expired_fix.config.deadline = Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .expect("one millisecond fits before the current instant"),
            );
            let expired_runtime = runtime_for(&expired_fix);
            let expired_engine = ScriptedResnetGradEngine::cooperative();
            let expired_ctx = ctx_of(&expired_fix, Some(&expired_engine));
            assert!(
                expired_fix
                    .graph
                    .try_gpu_warmup_bound_full(&expired_ctx, &expired_fix.bounds, &expired_runtime)
                    .is_none(),
                "expired warmup GPU work must fail closed"
            );
            assert_eq!(expired_engine.calls(), 0);
            assert!(expired_engine.deadline_writes().is_empty());

            let mut noncoop_fix = warmup_fixture();
            noncoop_fix.config.deadline = Some(Instant::now() + Duration::from_secs(30));
            let noncoop_runtime = runtime_for(&noncoop_fix);
            let noncoop_engine = ScriptedResnetGradEngine::default();
            let noncoop_ctx = ctx_of(&noncoop_fix, Some(&noncoop_engine));
            assert!(
                noncoop_fix
                    .graph
                    .try_gpu_warmup_bound_full(&noncoop_ctx, &noncoop_fix.bounds, &noncoop_runtime)
                    .is_none(),
                "noncooperative backend must use the CPU fallback"
            );
            assert_eq!(noncoop_engine.calls(), 0);

            let mut live_fix = warmup_fixture();
            let deadline = Instant::now() + Duration::from_secs(30);
            live_fix.config.deadline = Some(deadline);
            let live_runtime = runtime_for(&live_fix);
            let live_engine = ScriptedResnetGradEngine::cooperative();
            let live_ctx = ctx_of(&live_fix, Some(&live_engine));
            assert!(
                live_fix
                    .graph
                    .try_gpu_warmup_bound_full(&live_ctx, &live_fix.bounds, &live_runtime)
                    .is_some(),
                "cooperative backend should retain the GPU warmup path"
            );
            assert_eq!(live_engine.calls(), 1);
            assert_eq!(
                live_engine.deadline_writes(),
                vec![Some(deadline), None],
                "the exact backend deadline must be scoped to the resident fold"
            );
        });
    }

    #[test]
    fn margin_objective_replaces_identity_gradient_with_exact_spec_seed() {
        with_env_edits(|env| {
            env.set("NY_ROOT_ALPHA_GPU", "1");
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_EXTRACT_SKELETON");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA_GPU");
            env.remove("NY_ROOT_ALPHA_TRUE");
            env.remove("NY_BETA_GPU_PROBE");

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_secs(30));
            let runtime = runtime_for(&fix);
            let engine = ScriptedResnetGradEngine::cooperative();
            let ctx = ctx_of(&fix, Some(&engine));
            let local_grads = zero_local_grads(&fix);
            let (_bounds, mut cache) = fix
                .graph
                .try_gpu_warmup_bound_full(&ctx, &fix.bounds, &runtime)
                .expect("loop-top fold");
            engine.set_joint_widths_from_cache(&cache);
            cache.iter = 0;
            let spec: Vec<f32> = (0..ctx.output_dim)
                .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
                .collect();

            let gradients = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_grads,
                    Some(&cache),
                    0,
                    MarginGradientRequest::Binding(&spec),
                )
                .expect("binding-margin gradient");

            assert_eq!(
                engine.calls(),
                1,
                "the cached loop-top fold must not be recomputed"
            );
            assert_eq!(engine.joint_calls(), 1, "one direct-spec adjoint");
            assert_eq!(
                engine.joint_seeds(),
                vec![spec],
                "the adjoint seed must be the selected verification row, not identity"
            );
            assert_eq!(
                gradients.margin_dispatch,
                MarginGradientDispatch::JointDispatched {
                    source: MarginGradientJointSource::Resident
                }
            );
            assert!(
                gradients
                    .gradients
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite()),
                "a selected margin update may not admit non-finite gradients"
            );
            assert_eq!(
                gradients.gradients.len(),
                local_grads.len(),
                "the exact spec seed must map back to every runtime ReLU slot"
            );
        });
    }

    #[test]
    fn method_specific_joint_deadline_dispatches_without_global_deadline_capability() {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_BETA_GPU_PROBE");

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_secs(30));
            let runtime = runtime_for(&fix);
            let engine = ScriptedResnetGradEngine::joint_deadline_only();
            let ctx = ctx_of(&fix, Some(&engine));
            let (_, fold_names, _, _) = fix
                .graph
                .warmup_segments(&ctx, &fix.bounds, &runtime, "conv_out")
                .expect("method-specific extraction");
            engine.set_joint_widths(
                fold_names
                    .iter()
                    .map(|name| {
                        fix.relu_nodes
                            .iter()
                            .find_map(|(runtime_name, width)| {
                                (runtime_name == name).then_some(*width)
                            })
                            .expect("fold ReLU belongs to the runtime")
                    })
                    .collect(),
            );
            let supplied = sentinel_local_grads(&fix);
            let spec = vec![1.0; fix.output_dim];

            let result = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                )
                .expect("method-specific deadline lane");

            assert_eq!(
                result.margin_dispatch,
                MarginGradientDispatch::JointDispatched {
                    source: MarginGradientJointSource::Resident
                }
            );
            assert_eq!(
                engine.calls(),
                0,
                "the ordinary unbounded resident-gradient method must remain uncalled"
            );
            assert_eq!(engine.joint_calls(), 1);
            assert_eq!(engine.joint_seeds(), vec![spec]);
            assert!(
                engine.deadline_writes().is_empty(),
                "call-local joint cancellation must not mutate the global deadline slot"
            );
        });
    }

    #[test]
    fn method_specific_joint_deadline_error_keeps_truthful_supplied_fallback() {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_ROOT_ALPHA_GPU");
            env.remove("NY_BETA_GPU_PROBE");

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_millis(10));
            let runtime = runtime_for(&fix);
            let engine = ScriptedResnetGradEngine::joint_deadline_only_with_behavior(
                ScriptedJointBehavior::ScriptedTailPastDeadline,
            );
            let supplied = sentinel_local_grads(&fix);
            let spec = vec![1.0; fix.output_dim];

            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx_of(&fix, Some(&engine)),
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                ),
                &supplied,
                MarginGradientFallbackReason::DeadlineExpired,
                MarginGradientFallbackSource::SuppliedLocal,
            );
            assert_eq!(
                engine.calls(),
                0,
                "method-specific CUDA analogue must not enter ordinary resident work"
            );
            assert_eq!(engine.joint_calls(), 1);
            assert_eq!(
                engine.joint_work_units(),
                vec![0],
                "deadline telemetry must retain the cancellation that left the tail unexecuted"
            );
        });
    }

    /// #alpha-steering-proposal gating test 1 (seam): with NO
    /// verdict-authority engine (the Metal case), an armed BINDING objective
    /// dispatches the true binding-row adjoint through the dedicated proposal
    /// channel — `JointDispatched { source: WgpuProposal }` with the binding
    /// row as the adjoint seed — instead of dying at the authority filter.
    #[test]
    fn armed_binding_without_authority_dispatches_wgpu_proposal_channel() {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_ROOT_ALPHA_GPU");
            env.remove("NY_BETA_GPU_PROBE");

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_secs(30));
            let runtime = runtime_for(&fix);
            let steering =
                ScriptedResnetGradEngine::proposal_only_with_behavior(ScriptedJointBehavior::Valid);
            // Widths must match the runtime per-ReLU neuron counts so the
            // proposal maps back (same setup as the resident narrow-branch
            // test above).
            let probe_ctx = ctx_of(&fix, None);
            let (_, fold_names, _, _) = fix
                .graph
                .warmup_segments(&probe_ctx, &fix.bounds, &runtime, "conv_out")
                .expect("proposal extraction");
            steering.set_joint_widths(
                fold_names
                    .iter()
                    .map(|name| {
                        fix.relu_nodes
                            .iter()
                            .find_map(|(runtime_name, width)| {
                                (runtime_name == name).then_some(*width)
                            })
                            .expect("fold ReLU belongs to the runtime")
                    })
                    .collect(),
            );
            let ctx = ctx_with_steering(&fix, None, Some(&steering));
            let supplied = sentinel_local_grads(&fix);
            let spec = vec![1.0; fix.output_dim];

            let result = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                )
                .expect("proposal lane resolves");

            assert_eq!(
                result.margin_dispatch,
                MarginGradientDispatch::JointDispatched {
                    source: MarginGradientJointSource::WgpuProposal
                }
            );
            assert_eq!(steering.joint_calls(), 1, "one binding-row adjoint");
            assert_eq!(
                steering.joint_seeds(),
                vec![spec],
                "the adjoint seed must be the binding verification row"
            );
            assert!(
                result
                    .gradients
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite()),
                "proposal gradients pass the same finiteness moat as resident ones"
            );
            assert_eq!(result.gradients.len(), supplied.len());
            assert_ne!(
                grads_bits(&result.gradients),
                grads_bits(&supplied),
                "the joint proposal must actually replace the local rule"
            );
        });
    }

    /// #alpha-steering-proposal gating test 3a (bounds provably unconsumed):
    /// the proposal path dispatches ONLY the joint adjoint — an API whose
    /// return type is gradients — so no bound value from the steering engine
    /// exists to consume. Pinned mechanically: every bound-producing method
    /// of the scripted proposal engine counts invocations, and a successful
    /// proposal dispatch leaves all of them at zero.
    #[test]
    fn proposal_channel_never_invokes_bound_producing_methods() {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_ROOT_ALPHA_GPU");
            env.remove("NY_BETA_GPU_PROBE");

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_secs(30));
            let runtime = runtime_for(&fix);
            let steering =
                ScriptedResnetGradEngine::proposal_only_with_behavior(ScriptedJointBehavior::Valid);
            let probe_ctx = ctx_of(&fix, None);
            let (_, fold_names, _, _) = fix
                .graph
                .warmup_segments(&probe_ctx, &fix.bounds, &runtime, "conv_out")
                .expect("proposal extraction");
            steering.set_joint_widths(
                fold_names
                    .iter()
                    .map(|name| {
                        fix.relu_nodes
                            .iter()
                            .find_map(|(runtime_name, width)| {
                                (runtime_name == name).then_some(*width)
                            })
                            .expect("fold ReLU belongs to the runtime")
                    })
                    .collect(),
            );
            let ctx = ctx_with_steering(&fix, None, Some(&steering));
            let supplied = sentinel_local_grads(&fix);
            let spec = vec![1.0; fix.output_dim];

            let result = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                )
                .expect("proposal lane resolves");

            assert!(result.margin_dispatch.joint_dispatched());
            assert_eq!(
                steering.calls(),
                0,
                "the bound-producing resident fold (`crown_backward_gpu_resnet_sound_grad`) \
                 must never be invoked on the proposal channel — its NaN-poisonable bounds \
                 are unreachable by construction"
            );
            assert_eq!(steering.joint_calls(), 1);
        });
    }

    /// #alpha-steering-proposal default parity: channel absent ⇒ the armed
    /// no-authority iteration takes EXACTLY today's bounded local fallback
    /// (`resident_unavailable` / `supplied_local`, supplied gradients
    /// bit-identical).
    #[test]
    fn proposal_channel_absent_preserves_supplied_local_fallback() {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_ROOT_ALPHA_GPU");
            env.remove("NY_BETA_GPU_PROBE");

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_secs(30));
            let runtime = runtime_for(&fix);
            let supplied = sentinel_local_grads(&fix);
            let spec = vec![1.0; fix.output_dim];

            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx_of(&fix, None),
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                ),
                &supplied,
                MarginGradientFallbackReason::ResidentUnavailable,
                MarginGradientFallbackSource::SuppliedLocal,
            );
        });
    }

    /// #alpha-steering-proposal authority hygiene: the proposal channel is a
    /// LAST resort, consulted only when the authority filter yields nothing.
    /// With a verdict-authority engine present the resident lane dispatches
    /// and the steering handle is never touched; with the margin child
    /// disabled the proposal channel is likewise never consulted.
    #[test]
    fn proposal_channel_not_consulted_when_authority_present_or_child_disabled() {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_ROOT_ALPHA_GPU");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA_GPU");
            env.remove("NY_ROOT_ALPHA_TRUE");
            env.remove("NY_BETA_GPU_PROBE");

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_secs(30));
            let runtime = runtime_for(&fix);
            let steering =
                ScriptedResnetGradEngine::proposal_only_with_behavior(ScriptedJointBehavior::Valid);
            let authority = ScriptedResnetGradEngine::cooperative();
            let supplied = sentinel_local_grads(&fix);
            let spec = vec![1.0; fix.output_dim];

            // Authority present: resident lane wins, steering untouched.
            let ctx = ctx_with_steering(&fix, Some(&authority), Some(&steering));
            let result = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                )
                .expect("resident lane resolves");
            assert!(matches!(
                result.margin_dispatch,
                MarginGradientDispatch::JointDispatched {
                    source: MarginGradientJointSource::Resident
                } | MarginGradientDispatch::LocalFallback { .. }
            ));
            assert_eq!(
                steering.joint_calls(),
                0,
                "an available authority engine must shadow the proposal channel"
            );

            // Child disabled: the no-authority path must not consult the
            // channel either (byte-identical legacy dispatch).
            let ctx = ctx_with_steering(&fix, None, Some(&steering));
            let disabled = fix.graph.try_gpu_resnet_warmup_gradients(
                &ctx,
                &fix.bounds,
                &runtime,
                &supplied,
                None,
                0,
                MarginGradientRequest::Disabled,
            );
            assert!(
                disabled.is_none(),
                "child-disabled no-authority attempt keeps the legacy None"
            );
            assert_eq!(steering.joint_calls(), 0);
            assert_eq!(steering.calls(), 0);
        });
    }

    /// #alpha-steering-proposal gating test 3b (deadline contract at the
    /// proposal seam): a scripted tail past the deadline is cancelled
    /// cooperatively and reported as an explicit bounded fallback — no late
    /// gradient publication.
    #[test]
    fn proposal_channel_deadline_cancels_scripted_tail() {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_ROOT_ALPHA_GPU");
            env.remove("NY_BETA_GPU_PROBE");

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_millis(10));
            let runtime = runtime_for(&fix);
            let steering = ScriptedResnetGradEngine::proposal_only_with_behavior(
                ScriptedJointBehavior::ScriptedTailPastDeadline,
            );
            let supplied = sentinel_local_grads(&fix);
            let spec = vec![1.0; fix.output_dim];

            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx_with_steering(&fix, None, Some(&steering)),
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                ),
                &supplied,
                MarginGradientFallbackReason::DeadlineExpired,
                MarginGradientFallbackSource::SuppliedLocal,
            );
            assert_eq!(steering.joint_calls(), 1);
            assert_eq!(
                steering.joint_work_units(),
                vec![0],
                "the proposal adjoint must cancel cooperatively, leaving the tail unexecuted"
            );
        });
    }

    #[test]
    fn armed_margin_preflight_refusals_are_explicit_local_fallbacks() {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_BETA_GPU_PROBE");

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_secs(30));
            let runtime = runtime_for(&fix);
            let local_gradients = sentinel_local_grads(&fix);
            let spec = vec![1.0; fix.output_dim];

            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx_of(&fix, None),
                    &fix.bounds,
                    &runtime,
                    &local_gradients,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                ),
                &local_gradients,
                MarginGradientFallbackReason::ResidentUnavailable,
                MarginGradientFallbackSource::SuppliedLocal,
            );

            let engine = ScriptedResnetGradEngine::default();
            let ctx = ctx_of(&fix, Some(&engine));
            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx_of(&fix, None),
                    &fix.bounds,
                    &runtime,
                    &local_gradients,
                    None,
                    0,
                    MarginGradientRequest::NoBinding,
                ),
                &local_gradients,
                MarginGradientFallbackReason::NoBinding,
                MarginGradientFallbackSource::SuppliedLocal,
            );

            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_gradients,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec[..spec.len() - 1]),
                ),
                &local_gradients,
                MarginGradientFallbackReason::InvalidObjective,
                MarginGradientFallbackSource::SuppliedLocal,
            );
            let mut nonfinite_spec = spec.clone();
            nonfinite_spec[0] = f32::NAN;
            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_gradients,
                    None,
                    0,
                    MarginGradientRequest::Binding(&nonfinite_spec),
                ),
                &local_gradients,
                MarginGradientFallbackReason::InvalidObjective,
                MarginGradientFallbackSource::SuppliedLocal,
            );

            env.set("NY_RESNET_WARMUP_GPU", "0");
            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_gradients,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                ),
                &local_gradients,
                MarginGradientFallbackReason::ResidentUnavailable,
                MarginGradientFallbackSource::SuppliedLocal,
            );
            assert_eq!(engine.calls(), 0);
            assert_eq!(engine.joint_calls(), 0);
        });
    }

    #[test]
    fn armed_margin_deadline_refusals_never_enter_resident_work() {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_BETA_GPU_PROBE");

            let mut expired_fix = warmup_fixture();
            expired_fix.config.deadline = Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .expect("one millisecond fits before the current instant"),
            );
            let expired_runtime = runtime_for(&expired_fix);
            let expired_engine = ScriptedResnetGradEngine::cooperative();
            let expired_local = sentinel_local_grads(&expired_fix);
            let expired_spec = vec![1.0; expired_fix.output_dim];
            assert_local_margin_fallback(
                expired_fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx_of(&expired_fix, Some(&expired_engine)),
                    &expired_fix.bounds,
                    &expired_runtime,
                    &expired_local,
                    None,
                    0,
                    MarginGradientRequest::Binding(&expired_spec),
                ),
                &expired_local,
                MarginGradientFallbackReason::DeadlineExpired,
                MarginGradientFallbackSource::SuppliedLocal,
            );
            assert_eq!(expired_engine.calls(), 0);
            assert_eq!(expired_engine.joint_calls(), 0);
            assert!(expired_engine.deadline_writes().is_empty());

            let mut noncoop_fix = warmup_fixture();
            noncoop_fix.config.deadline = Some(Instant::now() + Duration::from_secs(30));
            let noncoop_runtime = runtime_for(&noncoop_fix);
            let noncoop_engine = ScriptedResnetGradEngine::default();
            let noncoop_local = sentinel_local_grads(&noncoop_fix);
            let noncoop_spec = vec![1.0; noncoop_fix.output_dim];
            assert_local_margin_fallback(
                noncoop_fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx_of(&noncoop_fix, Some(&noncoop_engine)),
                    &noncoop_fix.bounds,
                    &noncoop_runtime,
                    &noncoop_local,
                    None,
                    0,
                    MarginGradientRequest::Binding(&noncoop_spec),
                ),
                &noncoop_local,
                MarginGradientFallbackReason::ResidentUnavailable,
                MarginGradientFallbackSource::SuppliedLocal,
            );
            assert_eq!(noncoop_engine.calls(), 0);
            assert_eq!(noncoop_engine.joint_calls(), 0);
            assert!(noncoop_engine.deadline_writes().is_empty());
        });
    }

    #[test]
    fn armed_margin_joint_deadline_cancels_scripted_tail_and_uses_resident_local_rule() {
        with_env_edits(|env| {
            env.set("NY_ROOT_ALPHA_GPU", "1");
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_BETA_GPU_PROBE");

            let mut fix = warmup_fixture();
            let runtime = runtime_for(&fix);
            let engine = ScriptedResnetGradEngine::cooperative_with_joint_behavior(
                ScriptedJointBehavior::ScriptedTailPastDeadline,
            );
            let supplied_gradients = sentinel_local_grads(&fix);
            let spec: Vec<f32> = (0..fix.output_dim)
                .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            let (cache, disabled_gradients) = {
                let ctx = ctx_of(&fix, Some(&engine));
                let (_bound, mut cache) = fix
                    .graph
                    .try_gpu_warmup_bound_full(&ctx, &fix.bounds, &runtime)
                    .expect("scripted loop-top fold");
                engine.set_joint_widths_from_cache(&cache);
                cache.iter = 7;
                let disabled = fix
                    .graph
                    .try_gpu_resnet_warmup_gradients(
                        &ctx,
                        &fix.bounds,
                        &runtime,
                        &supplied_gradients,
                        Some(&cache),
                        7,
                        MarginGradientRequest::Disabled,
                    )
                    .expect("child-disabled resident local rule");
                (cache, disabled.gradients)
            };

            fix.config.deadline = Some(Instant::now() + Duration::from_millis(10));
            let deadline_ctx = ctx_of(&fix, Some(&engine));
            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &deadline_ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied_gradients,
                    Some(&cache),
                    7,
                    MarginGradientRequest::Binding(&spec),
                ),
                &disabled_gradients,
                MarginGradientFallbackReason::DeadlineExpired,
                MarginGradientFallbackSource::ResidentLocalRule,
            );
            assert_eq!(engine.calls(), 1, "the loop-top cache must be reused");
            assert_eq!(
                engine.joint_calls(),
                1,
                "the joint must dispatch before expiry"
            );
            assert_eq!(
                engine.joint_work_units(),
                vec![0],
                "cooperative polling must leave the scripted joint tail unexecuted"
            );
        });
    }

    #[test]
    fn armed_margin_joint_and_mapping_refusals_are_bounded_and_auditable() {
        with_env_edits(|env| {
            env.set("NY_ROOT_ALPHA_GPU", "1");
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_BETA_GPU_PROBE");

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_secs(30));
            let supplied_gradients = sentinel_local_grads(&fix);
            let spec: Vec<f32> = (0..fix.output_dim)
                .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            let cases = [
                (
                    ScriptedJointBehavior::Error,
                    MarginGradientFallbackReason::JointUnavailable,
                ),
                (
                    ScriptedJointBehavior::WrongReluCount,
                    MarginGradientFallbackReason::JointUnavailable,
                ),
                (
                    ScriptedJointBehavior::NonFinite,
                    MarginGradientFallbackReason::NonFiniteGradient,
                ),
                (
                    ScriptedJointBehavior::WrongNeuronCount,
                    MarginGradientFallbackReason::MappingMismatch,
                ),
            ];

            for (behavior, expected_reason) in cases {
                let runtime = runtime_for(&fix);
                let engine = ScriptedResnetGradEngine::cooperative_with_joint_behavior(behavior);
                let ctx = ctx_of(&fix, Some(&engine));
                let (_bound, mut cache) = fix
                    .graph
                    .try_gpu_warmup_bound_full(&ctx, &fix.bounds, &runtime)
                    .expect("scripted loop-top fold");
                engine.set_joint_widths_from_cache(&cache);
                cache.iter = 3;
                let disabled = fix
                    .graph
                    .try_gpu_resnet_warmup_gradients(
                        &ctx,
                        &fix.bounds,
                        &runtime,
                        &supplied_gradients,
                        Some(&cache),
                        3,
                        MarginGradientRequest::Disabled,
                    )
                    .expect("child-disabled resident local rule");
                assert_eq!(
                    disabled.margin_dispatch,
                    MarginGradientDispatch::NotRequested
                );
                assert_ne!(
                    grads_bits(&disabled.gradients),
                    grads_bits(&supplied_gradients),
                    "the parity oracle must distinguish resident-local from supplied-local"
                );
                assert!(
                    disabled
                        .gradients
                        .iter()
                        .flatten()
                        .any(|value| *value != 0.0),
                    "the child-disabled resident oracle must be nonzero"
                );
                if matches!(behavior, ScriptedJointBehavior::NonFinite) {
                    for pre_lower in &mut cache.pre_lowers {
                        pre_lower.fill(0.0);
                    }
                }
                assert_local_margin_fallback(
                    fix.graph.try_gpu_resnet_warmup_gradients(
                        &ctx,
                        &fix.bounds,
                        &runtime,
                        &supplied_gradients,
                        Some(&cache),
                        3,
                        MarginGradientRequest::Binding(&spec),
                    ),
                    &disabled.gradients,
                    expected_reason,
                    MarginGradientFallbackSource::ResidentLocalRule,
                );
                assert_eq!(engine.calls(), 1, "the cached fold must be reused");
                assert_eq!(engine.joint_calls(), 1);
            }

            let runtime = runtime_for(&fix);
            let engine = ScriptedResnetGradEngine::cooperative();
            let ctx = ctx_of(&fix, Some(&engine));
            let (_bound, mut missing_mask_cache) = fix
                .graph
                .try_gpu_warmup_bound_full(&ctx, &fix.bounds, &runtime)
                .expect("scripted loop-top fold");
            engine.set_joint_widths_from_cache(&missing_mask_cache);
            missing_mask_cache.iter = 4;
            let disabled_for_missing_mask = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied_gradients,
                    Some(&missing_mask_cache),
                    4,
                    MarginGradientRequest::Disabled,
                )
                .expect("child-disabled resident local rule");
            missing_mask_cache.pre_lowers.pop();
            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied_gradients,
                    Some(&missing_mask_cache),
                    4,
                    MarginGradientRequest::Binding(&spec),
                ),
                &disabled_for_missing_mask.gradients,
                MarginGradientFallbackReason::MappingMismatch,
                MarginGradientFallbackSource::ResidentLocalRule,
            );

            let runtime = runtime_for(&fix);
            let engine = ScriptedResnetGradEngine::cooperative();
            let ctx = ctx_of(&fix, Some(&engine));
            let (_bound, mut cache) = fix
                .graph
                .try_gpu_warmup_bound_full(&ctx, &fix.bounds, &runtime)
                .expect("scripted loop-top fold");
            cache.iter = 4;
            let disabled = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied_gradients,
                    Some(&cache),
                    4,
                    MarginGradientRequest::Disabled,
                )
                .expect("child-disabled resident local rule");
            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied_gradients,
                    Some(&cache),
                    4,
                    MarginGradientRequest::NoBinding,
                ),
                &disabled.gradients,
                MarginGradientFallbackReason::NoBinding,
                MarginGradientFallbackSource::ResidentLocalRule,
            );
            assert_eq!(engine.joint_calls(), 0, "no binding must not run the joint");

            let runtime = runtime_for(&fix);
            let engine = ScriptedResnetGradEngine::cooperative();
            let ctx = ctx_of(&fix, Some(&engine));
            let (_bound, mut cache) = fix
                .graph
                .try_gpu_warmup_bound_full(&ctx, &fix.bounds, &runtime)
                .expect("scripted loop-top fold");
            engine.set_joint_widths_from_cache(&cache);
            cache.iter = 5;
            cache.relu_names[0] = "missing-runtime-relu".to_string();
            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied_gradients,
                    Some(&cache),
                    5,
                    MarginGradientRequest::Binding(&spec),
                ),
                &supplied_gradients,
                MarginGradientFallbackReason::MappingMismatch,
                MarginGradientFallbackSource::SuppliedLocal,
            );
            assert_eq!(engine.calls(), 1);
            assert_eq!(engine.joint_calls(), 1);
        });
    }

    #[test]
    fn armed_margin_dispatcher_moat_never_runs_analytic_chain_on_resident_miss() {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_BETA_GPU_PROBE");

            let fix = warmup_fixture();
            let ctx = ctx_of(&fix, None);
            assert!(matches!(
                ctx.config.gradient_method,
                GradientMethod::AnalyticChain
            ));
            let mut runtime = runtime_for(&fix);
            let mut bilinear_alphas = HashMap::new();
            let mut mul_binary_alphas = HashMap::new();
            let local_gradients = sentinel_local_grads(&fix);
            let local_gradients_upper = sentinel_local_grads(&fix);
            let spec = vec![1.0; fix.output_dim];

            let result = fix
                .graph
                .compute_dag_gradients(
                    &ctx,
                    &fix.bounds,
                    &mut runtime,
                    &mut bilinear_alphas,
                    &mut mul_binary_alphas,
                    &local_gradients,
                    &local_gradients_upper,
                    1e-3,
                    0,
                    None,
                    MarginGradientRequest::Binding(&spec),
                )
                .expect("armed resident miss must resolve without CPU chain replay");

            assert_eq!(
                result.margin_dispatch,
                MarginGradientDispatch::LocalFallback {
                    reason: MarginGradientFallbackReason::ResidentUnavailable,
                    source: MarginGradientFallbackSource::SuppliedLocal,
                }
            );
            assert_eq!(
                grads_bits(&result.numerical_gradients),
                grads_bits(&local_gradients)
            );
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
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_grads,
                    None,
                    4,
                    MarginGradientRequest::Disabled,
                )
                .expect("fresh gradient fold succeeds");
            let calls_after_fresh = engine.calls();
            assert_eq!(
                calls_after_fresh, 3,
                "fresh gradient fold re-runs the kernel"
            );
            assert!(
                fresh.gradients.iter().flatten().any(|&g| g != 0.0),
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
                    MarginGradientRequest::Disabled,
                )
                .expect("cached gradient path succeeds");
            assert_eq!(
                engine.calls(),
                calls_after_fresh,
                "cache hit must not re-run the kernel (one fold per iteration)"
            );
            assert_eq!(
                grads_bits(&fresh.gradients),
                grads_bits(&cached.gradients),
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
                    MarginGradientRequest::Disabled,
                )
                .expect("second cached run succeeds");
            assert_eq!(engine.calls(), calls_after_fresh);
            assert_eq!(
                grads_bits(&cached.gradients),
                grads_bits(&cached_again.gradients)
            );
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
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &local_grads,
                    None,
                    2,
                    MarginGradientRequest::Disabled,
                )
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
                    MarginGradientRequest::Disabled,
                )
                .expect("cache hit succeeds");
            assert!(
                hit.gradients.iter().flatten().all(|&g| g == 42.0),
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
                    MarginGradientRequest::Disabled,
                )
                .expect("refresh-invalidated path succeeds");
            assert_eq!(
                engine.calls(),
                calls_before + 1,
                "refresh_fired must force a fresh kernel fold"
            );
            assert_eq!(
                grads_bits(&after_refresh.gradients),
                grads_bits(&fresh.gradients)
            );
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
                    MarginGradientRequest::Disabled,
                )
                .expect("stale-iter path succeeds");
            assert_eq!(
                engine.calls(),
                calls_before + 2,
                "a stale iter must force a fresh kernel fold"
            );
            assert_eq!(
                grads_bits(&stale_iter.gradients),
                grads_bits(&fresh.gradients)
            );

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
                    MarginGradientRequest::Disabled,
                )
                .expect("gate-off path succeeds");
            assert_eq!(
                engine.calls(),
                calls_before + 3,
                "with the gate off the cache must be ignored"
            );
            assert_eq!(
                grads_bits(&gate_off.gradients),
                grads_bits(&fresh.gradients)
            );
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

    #[test]
    fn conflicting_alpha_policies_keep_exact_child_disabled_loop() {
        let run = |child: bool, multiobj: bool, root_true: bool| {
            with_env_edits(|env| {
                env.remove("NY_ROOT_ALPHA_GPU");
                env.set("NY_RESNET_WARMUP_GPU", "1");
                env.set("NY_ROOT_ALPHA_MARGIN", "1");
                if child {
                    env.set("NY_ROOT_ALPHA_MARGIN_GRADIENT", "1");
                } else {
                    env.remove("NY_ROOT_ALPHA_MARGIN_GRADIENT");
                }
                if multiobj {
                    env.set("NY_MULTIOBJ_JOINT_ALPHA", "1");
                } else {
                    env.remove("NY_MULTIOBJ_JOINT_ALPHA");
                }
                env.remove("NY_MULTIOBJ_JOINT_ALPHA_GPU");
                if root_true {
                    env.set("NY_ROOT_ALPHA_TRUE", "1");
                } else {
                    env.remove("NY_ROOT_ALPHA_TRUE");
                }
                env.remove("NY_BETA_GPU_PROBE");

                let fix = warmup_fixture();
                let engine = ScriptedResnetGradEngine::default();
                let mut objective = vec![0.0; fix.output_dim];
                objective[0] = 1.0;
                let config = AlphaCrownConfig {
                    iterations: 2,
                    spec_ascent: AlphaSpecAscent::new(vec![AlphaSpecEarlyExit {
                        objective,
                        threshold: 100.0,
                        verify_upper_bound: false,
                    }]),
                    ..Default::default()
                };
                fix.graph
                    .propagate_dag_alpha_crown_with_config_and_engine(
                        &fix.input,
                        &config,
                        Some(&engine),
                    )
                    .expect("conflicting-policy DAG alpha loop")
            })
        };

        for (multiobj, root_true, label) in [
            (true, false, "NY_MULTIOBJ_JOINT_ALPHA"),
            (false, true, "NY_ROOT_ALPHA_TRUE"),
            (true, true, "both pre-existing alpha policies"),
        ] {
            let child_disabled = run(false, multiobj, root_true);
            let child_coarmed = run(true, multiobj, root_true);
            assert_eq!(
                tensor_bits(&child_coarmed),
                tensor_bits(&child_disabled),
                "{label}: ineligible margin child must preserve the existing policy bit-for-bit"
            );
        }
    }

    /// #alpha-steering-proposal loop-level fixture: armed margin loop on the
    /// conv fixture with a scripted proposal engine injected on the steering
    /// channel (no authority engine — the Metal shape). Returns the published
    /// bounds and the scripted engine (for dispatch-count asserts).
    fn run_margin_loop_with_steering(
        behavior: Option<ScriptedJointBehavior>,
        iterations: usize,
    ) -> (
        BoundedTensor,
        Option<std::sync::Arc<ScriptedResnetGradEngine>>,
    ) {
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.set("NY_ROOT_ALPHA_MARGIN", "1");
            env.set("NY_ROOT_ALPHA_MARGIN_GRADIENT", "1");
            env.remove("NY_ROOT_ALPHA_MARGIN_HINGE");
            env.remove("NY_ROOT_ALPHA_GPU");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA");
            env.remove("NY_MULTIOBJ_JOINT_ALPHA_GPU");
            env.remove("NY_ROOT_ALPHA_TRUE");
            env.remove("NY_BETA_GPU_PROBE");
            env.remove("NY_ALPHA_DIVERGENCE_BAIL");

            let mut fix = warmup_fixture();
            let mut objective = vec![0.0; fix.output_dim];
            objective[0] = 1.0;
            fix.config = AlphaCrownConfig {
                iterations,
                deadline: Some(Instant::now() + Duration::from_mins(1)),
                spec_ascent: AlphaSpecAscent::new(vec![AlphaSpecEarlyExit {
                    objective,
                    threshold: 100.0,
                    verify_upper_bound: false,
                }]),
                ..Default::default()
            };

            let steering = behavior.map(|behavior| {
                let runtime = runtime_for(&fix);
                let engine = ScriptedResnetGradEngine::proposal_only_with_behavior(behavior);
                let probe_ctx = ctx_of(&fix, None);
                let (_, fold_names, _, _) = fix
                    .graph
                    .warmup_segments(&probe_ctx, &fix.bounds, &runtime, "conv_out")
                    .expect("proposal extraction");
                engine.set_joint_widths(
                    fold_names
                        .iter()
                        .map(|name| {
                            fix.relu_nodes
                                .iter()
                                .find_map(|(runtime_name, width)| {
                                    (runtime_name == name).then_some(*width)
                                })
                                .expect("fold ReLU belongs to the runtime")
                        })
                        .collect(),
                );
                std::sync::Arc::new(engine)
            });

            let run = || {
                fix.graph
                    .propagate_dag_alpha_crown_with_config_and_engine(&fix.input, &fix.config, None)
                    .expect("armed margin DAG alpha loop")
            };
            let bounds = match steering.as_ref() {
                Some(engine) => crate::alpha_gradient_steering::with_test_steering(
                    engine.clone() as std::sync::Arc<dyn ny_core::GemmEngine>,
                    run,
                ),
                None => run(),
            };
            (bounds, steering)
        })
    }

    /// #alpha-steering-proposal gating test 2 (loop level): an adversarially
    /// WRONG proposal gradient — the exact sign-flip of the scripted joint
    /// adjoint — CANNOT regress the published bound below the certified CROWN
    /// baseline. Element-wise best-state retention
    /// (`update_elementwise_best_bounds`, seeded from the pre-loop CROWN
    /// bound) is the loop-level moat that makes proposal-grade gradients
    /// sound by construction.
    #[test]
    fn sign_flipped_proposal_gradient_cannot_regress_published_bound() {
        // Baseline: the pre-loop certified CROWN bound (zero iterations —
        // best-state retention seeds from exactly this enclosure).
        let (baseline, _) = run_margin_loop_with_steering(None, 0);
        let (poisoned, steering) =
            run_margin_loop_with_steering(Some(ScriptedJointBehavior::SignFlipped), 3);
        let steering = steering.expect("steering arm");
        assert!(
            steering.joint_calls() >= 1,
            "the adversarial proposal must actually have been dispatched"
        );
        assert_eq!(
            steering.calls(),
            0,
            "no bound-producing steering method may run at loop level either"
        );
        for (index, (poisoned_lower, baseline_lower)) in poisoned
            .lower()
            .iter()
            .zip(baseline.lower().iter())
            .enumerate()
        {
            assert!(
                poisoned_lower >= baseline_lower,
                "published lower bound regressed at {index}: {poisoned_lower} < {baseline_lower}"
            );
        }
        for (index, (poisoned_upper, baseline_upper)) in poisoned
            .upper()
            .iter()
            .zip(baseline.upper().iter())
            .enumerate()
        {
            assert!(
                poisoned_upper <= baseline_upper,
                "published upper bound regressed at {index}: {poisoned_upper} > {baseline_upper}"
            );
        }
    }

    /// #alpha-steering-proposal gating test 3c (poison, loop level): a
    /// NaN-poisoned proposal is a typed `nonfinite_gradient` refusal inside
    /// the fold moat, and the published result is BIT-IDENTICAL to the
    /// channel-absent run — the poisoned values reach neither α nor any
    /// bound.
    #[test]
    fn nan_poisoned_proposal_leaves_published_result_bit_identical() {
        let (absent, _) = run_margin_loop_with_steering(None, 3);
        let (poisoned, steering) =
            run_margin_loop_with_steering(Some(ScriptedJointBehavior::NonFinite), 3);
        let steering = steering.expect("steering arm");
        assert!(
            steering.joint_calls() >= 1,
            "the poisoned proposal must actually have been consulted"
        );
        assert_eq!(
            tensor_bits(&poisoned),
            tensor_bits(&absent),
            "a refused NaN proposal must leave the run byte-identical to no channel at all"
        );
    }

    /// VERIFIER poison variant (adversarial re-derivation of gating test 3c):
    /// poison a DIFFERENT layer than the author's test — the DEEPEST ReLU's
    /// last neuron instead of the first ReLU's first neuron — and a different
    /// non-finite class (`-inf` instead of NaN). The finiteness moat in
    /// `joint_alpha_grads_fold_gpu_with_deadline` scans every value of every
    /// per-ReLU gradient before masking, so the refusal and the published
    /// result must be exactly as bit-identical as the head-poisoned case.
    #[test]
    fn neg_inf_poisoned_deepest_layer_proposal_leaves_published_result_bit_identical() {
        let (absent, _) = run_margin_loop_with_steering(None, 3);
        let (poisoned, steering) =
            run_margin_loop_with_steering(Some(ScriptedJointBehavior::NegInfDeepLayer), 3);
        let steering = steering.expect("steering arm");
        assert!(
            steering.joint_calls() >= 1,
            "the deep-poisoned proposal must actually have been consulted"
        );
        assert_eq!(
            steering.calls(),
            0,
            "no bound-producing steering method may run under the deep poison either"
        );
        assert_eq!(
            tensor_bits(&poisoned),
            tensor_bits(&absent),
            "a -inf poisoned DEEPEST-layer proposal must leave the run byte-identical \
             to no channel at all — the moat must scan past the first layer"
        );
    }

    /// VERIFIER fixture for the direction gate: the conv fixture WITHOUT the
    /// MaxPool2d node and WITH merged lower/upper alphas. Rationale
    /// (adversarial finding, see verify report): the joint adjoint — CUDA
    /// twin included (ny-cuda joint_alpha.rs) — typed-refuses
    /// `ActivationReluDualAlpha` and `MaxPool2d` segments, and the stock
    /// `warmup_fixture()`+`runtime_for()` produce BOTH (maxpool node; mk_alpha
    /// 0.35 != 0.65), so the proposal channel can never dispatch a REAL
    /// adjoint on it. The armed production loop applies identical
    /// lower/upper updates to alphas that start merged, so the real lane
    /// presents merged-alpha `Activation` segments; this fixture reproduces
    /// that shape.
    #[cfg(feature = "gpu-tests")]
    fn joint_dispatchable_fixture() -> (WarmupFixture, DagAlphaRuntimeState) {
        use crate::network::graph_alpha::resnet_skeleton::test_support::{add, conv, lcg};
        let mut rng = lcg(0x51EE_D00D_F00D);
        let mut graph = GraphNetwork::new();
        graph.add_node(conv(
            &mut rng,
            "conv0",
            NETWORK_INPUT,
            (2, 4),
            3,
            1,
            1,
            true,
        ));
        graph.add_node(relu("relu0", "conv0"));
        graph.add_node(conv(&mut rng, "b1c1", "relu0", (4, 4), 3, 1, 1, false));
        graph.add_node(relu("b1r1", "b1c1"));
        graph.add_node(conv(&mut rng, "b1c2", "b1r1", (4, 4), 3, 1, 1, true));
        graph.add_node(add("add1", "b1c2", "relu0"));
        graph.add_node(relu("relu_out", "add1"));
        graph.add_node(conv(
            &mut rng,
            "conv_out",
            "relu_out",
            (4, 2),
            1,
            1,
            0,
            true,
        ));
        graph.set_output("conv_out");

        let relus = ["relu0", "b1r1", "relu_out"];
        let input = box_input(&[2, 6, 6], -1.0, 1.0);
        let bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let exec_order: Vec<String> = graph.node_order.clone();
        let relu_nodes: Vec<(String, usize)> = relus
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
        // MERGED alphas (lo == hi): the extraction stays on the plain
        // `Activation` variant the joint adjoint supports.
        let alpha = mk_alpha(&graph, &bounds, &relus, 0.45, 0.45);
        let runtime =
            DagAlphaRuntimeState::new(alpha, None, relus.iter().map(|s| s.to_string()).collect());
        (
            WarmupFixture {
                graph,
                input,
                bounds,
                exec_order,
                relu_nodes,
                output_dim,
                input_dim,
                config: AlphaCrownConfig::default(),
            },
            runtime,
        )
    }

    /// CHANNEL-SHARED alpha state for the fixture graph (#channel-alpha-grad):
    /// `add_relu_node(.., channel_only_alpha=true)` — the production
    /// `full_conv_alpha: false` wiring (`propagate_dag/init.rs:162`) — with
    /// merged lower/upper values so extraction stays on the plain
    /// `Activation` segment variant.
    fn mk_channel_alpha(
        graph: &GraphNetwork,
        bounds: &HashMap<String, BoundedTensor>,
        relus: &[&str],
        value: f32,
    ) -> GraphAlphaState {
        let mut ga = GraphAlphaState::new();
        for name in relus {
            let pre_name = graph
                .nodes
                .get(*name)
                .expect("fixture relu")
                .inputs
                .first()
                .expect("relu input")
                .clone();
            let pre = bounds.get(&pre_name).expect("pre bounds");
            ga.add_relu_node(name, pre, true).expect("add relu node");
            assert!(
                ga.spatial_shape(name).is_some(),
                "fixture ReLU '{name}' must take the channel-only path"
            );
            if let Some((l, u)) = ga.relu_alpha_pair_mut(name) {
                l.fill(value);
                u.fill(value);
            }
        }
        ga
    }

    /// #channel-alpha-grad mapper gate: per-neuron (C·H·W) fold gradients
    /// against a channel-shared (C) supplied layout reduce by the channel sum
    /// `dL/dα_c = Σ_{h,w} dL/dα_{c,h,w}` in BOTH mapping twins, the reduced
    /// layout is exactly what `update_all_alphas` consumes
    /// (`reduce_gradient` is the identity on it), and genuinely
    /// irreconcilable widths still refuse (`None` / `Ok(None)`).
    #[test]
    fn mapper_reduces_per_neuron_fold_gradients_to_channel_alpha_layout() {
        let fix = warmup_fixture();
        let alpha = mk_channel_alpha(&fix.graph, &fix.bounds, &CONV_FIXTURE_RELUS, 0.45);
        let ctx = ctx_of(&fix, None);
        let relu_names: Vec<String> = CONV_FIXTURE_RELUS.iter().map(|s| s.to_string()).collect();

        // Supplied local gradients at ALPHA width (the armed lane's reality:
        // backward/nonlinear.rs writes reduce_gradient output, length C).
        let supplied: Vec<Array1<f32>> = relu_names
            .iter()
            .map(|name| Array1::from_elem(alpha.alpha(name).expect("alpha").len(), 777.0))
            .collect();

        // Per-neuron fold gradients with a channel/spatial-legible pattern.
        let fold: Vec<Vec<f32>> = relu_names
            .iter()
            .zip(fix.relu_nodes.iter())
            .map(|(name, (_, width))| {
                let shape = alpha.spatial_shape(name).expect("geometry");
                let spatial: usize = shape[1..].iter().product();
                assert_eq!(shape[0] * spatial, *width, "fixture geometry sanity");
                (0..*width)
                    .map(|i| (i / spatial) as f32 * 10.0 + (i % spatial) as f32 * 0.5)
                    .collect()
            })
            .collect();

        let expected: Vec<Vec<f32>> = fold
            .iter()
            .zip(relu_names.iter())
            .map(|(g, name)| {
                let shape = alpha.spatial_shape(name).expect("geometry");
                let spatial: usize = shape[1..].iter().product();
                (0..shape[0])
                    .map(|c| g[c * spatial..(c + 1) * spatial].iter().sum())
                    .collect()
            })
            .collect();

        let mapped =
            GraphNetwork::map_warmup_fold_gradients(&ctx, &alpha, &supplied, &relu_names, &fold)
                .expect("channel-shared layouts must now map");
        let deadline = Instant::now() + Duration::from_mins(1);
        let mapped_deadline = GraphNetwork::map_warmup_fold_gradients_with_deadline(
            &ctx,
            &alpha,
            &supplied,
            &relu_names,
            &fold,
            deadline,
        )
        .expect("deadline mapping succeeds")
        .expect("deadline mapping reduces too");

        for (index, name) in relu_names.iter().enumerate() {
            let alpha_len = alpha.alpha(name).expect("alpha").len();
            assert_eq!(
                mapped[index].len(),
                alpha_len,
                "alpha-width output at '{name}'"
            );
            assert_eq!(
                mapped[index].as_slice().expect("contiguous"),
                expected[index].as_slice(),
                "channel sums at '{name}'"
            );
            // update_all_alphas expectation: its reduce_gradient is the
            // identity on the reduced layout (bitwise), and the width matches
            // the alpha vector the optimizer updates.
            let re_reduced = alpha.reduce_gradient(name, &mapped[index]);
            assert_eq!(
                grads_bits(std::slice::from_ref(&re_reduced)),
                grads_bits(std::slice::from_ref(&mapped[index])),
                "reduce_gradient must be a no-op on the mapper's output at '{name}'"
            );
        }
        assert_eq!(
            grads_bits(&mapped_deadline),
            grads_bits(&mapped),
            "both mapping twins must reduce identically"
        );

        // Irreconcilable: a fold gradient that is neither the alpha width nor
        // C·H·W must refuse in both twins.
        let mut bad = fold.clone();
        bad[1].pop();
        assert!(
            GraphNetwork::map_warmup_fold_gradients(&ctx, &alpha, &supplied, &relu_names, &bad)
                .is_none(),
            "truncated per-neuron gradient must refuse"
        );
        assert_eq!(
            GraphNetwork::map_warmup_fold_gradients_with_deadline(
                &ctx,
                &alpha,
                &supplied,
                &relu_names,
                &bad,
                Instant::now() + Duration::from_mins(1),
            )
            .expect("no deadline error"),
            None,
            "deadline twin must refuse the same layout"
        );

        // A supplied width that is not the stored alpha width must refuse too
        // (the geometry key is the MEASURED stored alpha, not divisibility).
        let mut bad_supplied = supplied.clone();
        bad_supplied[0] = Array1::from_elem(supplied[0].len() + 1, 777.0);
        assert!(
            GraphNetwork::map_warmup_fold_gradients(
                &ctx,
                &alpha,
                &bad_supplied,
                &relu_names,
                &fold
            )
            .is_none(),
            "a target width that is not the stored alpha width must refuse"
        );
    }

    /// VERIFIER direction-fidelity gate: the gradient the seam actually hands
    /// the loop — real `ny_gpu::GradientSteeringDevice` wgpu joint adjoint,
    /// through the full extraction + dispatch + mapping path — must point in
    /// the SAME direction as the independent AnalyticChain binding-row replay
    /// (`binding_row_true_alpha_grads`), the FD-proven true
    /// d(binding-row lower bound)/dα of the CERTIFIED CPU fold. This is the
    /// cross-implementation check the resident FD test cannot give (it
    /// compares the GPU adjoint against FD of the GPU's own bound and against
    /// the CPU twin of the same fold semantics).
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn proposal_joint_gradient_direction_matches_binding_row_replay() {
        assert!(
            ny_gpu::wgpu_adapter_available(),
            "gpu-tests requires a usable WGPU adapter"
        );
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_ROOT_ALPHA_GPU");
            env.remove("NY_BETA_GPU_PROBE");

            let (mut fix, runtime) = joint_dispatchable_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_mins(2));
            let steering =
                ny_gpu::GradientSteeringDevice::new_wgpu().expect("adapter probed available");
            let ctx = ctx_with_steering(&fix, None, Some(&steering));
            let supplied = sentinel_local_grads(&fix);
            // Binding row 0 as a one-hot spec row: the same row is addressable
            // in the identity-seeded replay fold below.
            let binding_row = 0usize;
            let mut spec = vec![0.0f32; fix.output_dim];
            spec[binding_row] = 1.0;

            let result = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                )
                .expect("proposal lane resolves");
            assert_eq!(
                result.margin_dispatch,
                MarginGradientDispatch::JointDispatched {
                    source: MarginGradientJointSource::WgpuProposal
                },
                "the real wgpu steering device must dispatch through the proposal channel"
            );

            // Independent oracle: AnalyticChain intermediates fold at the SAME
            // alpha iterate, then the binding-row replay (FD-correct against
            // the certified CPU fold, binding_row_replay/tests.rs).
            let mut grads_lower: Vec<Array1<f32>> = zero_local_grads(&fix);
            let mut grads_upper: Vec<Array1<f32>> = zero_local_grads(&fix);
            let (_, intermediate) = fix
                .graph
                .dag_alpha_backward_pass_with_intermediates(
                    &fix.input,
                    &fix.bounds,
                    &fix.exec_order,
                    fix.output_dim,
                    fix.input_dim,
                    runtime.relu_name_to_idx(),
                    runtime.graph(),
                    None,
                    &mut grads_lower,
                    &mut grads_upper,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("AnalyticChain intermediates fold");
            let replay = fix
                .graph
                .binding_row_true_alpha_grads(
                    &fix.input,
                    runtime.graph(),
                    &intermediate,
                    binding_row,
                )
                .expect("binding-row replay");

            // Compare per ReLU, in runtime order; every fixture ReLU must have
            // been replaced by the fold (no sentinel survivors) and replayed.
            let mut dot = 0.0f64;
            let mut seam_sq = 0.0f64;
            let mut replay_sq = 0.0f64;
            for (index, (name, _)) in fix.relu_nodes.iter().enumerate() {
                let seam = &result.gradients[index];
                assert_ne!(
                    grads_bits(std::slice::from_ref(seam)),
                    grads_bits(std::slice::from_ref(&supplied[index])),
                    "ReLU '{name}' must carry the joint proposal, not the sentinel local rule"
                );
                let oracle = replay
                    .grads
                    .get(name)
                    .unwrap_or_else(|| panic!("replay gradient for ReLU '{name}'"));
                assert_eq!(seam.len(), oracle.len(), "width parity at '{name}'");
                for (s, o) in seam.iter().zip(oracle.iter()) {
                    dot += f64::from(*s) * f64::from(*o);
                    seam_sq += f64::from(*s) * f64::from(*s);
                    replay_sq += f64::from(*o) * f64::from(*o);
                }
            }
            assert!(
                replay_sq > 0.0,
                "degenerate fixture: the replay oracle found no alpha sensitivity at all"
            );
            assert!(
                seam_sq > 0.0,
                "degenerate seam output: the proposal gradient is identically zero"
            );
            let cosine = dot / (seam_sq.sqrt() * replay_sq.sqrt());
            eprintln!(
                "[proposal-vs-replay] cosine={cosine:.6} |seam|={:.4e} |replay|={:.4e}",
                seam_sq.sqrt(),
                replay_sq.sqrt()
            );
            assert!(
                cosine > 0.98,
                "proposal-channel gradient must point where the AnalyticChain binding-row \
                 replay points (cosine {cosine})"
            );
            // MEASURED at introduction: cosine 1.000000 with |seam| == |replay|
            // to 5 significant digits — the seam gradient IS the replay
            // gradient, magnitude included; pin that with a relative-L2 gate.
            let mut diff_sq = 0.0f64;
            for (index, (name, _)) in fix.relu_nodes.iter().enumerate() {
                let seam = &result.gradients[index];
                let oracle = &replay.grads[name];
                for (s, o) in seam.iter().zip(oracle.iter()) {
                    let diff = f64::from(*s) - f64::from(*o);
                    diff_sq += diff * diff;
                }
            }
            let rel_l2 = (diff_sq / replay_sq.max(1e-30)).sqrt();
            assert!(
                rel_l2 < 5e-2,
                "proposal-channel gradient must match the AnalyticChain replay in magnitude \
                 too (rel_l2 {rel_l2})"
            );
        });
    }

    /// #channel-alpha-grad end-to-end cross-oracle gate — the exact
    /// production layout that failed the 2026-08-01 release A/B
    /// (`dispatch=local_fallback reason=mapping_mismatch`, arbitration doc
    /// §5): CHANNEL-SHARED α (`full_conv_alpha: false`), real wgpu joint
    /// adjoint emitting per-neuron (C·H·W) gradients (the kernel itself never
    /// speaks channel α — extraction expands, resnet_decompose.rs:588; the
    /// reduction is host-side in the mapper), mapped through the seam into
    /// the ALPHA-width (C) layout, versus the independent binding-row replay
    /// at the same iterate. The dispatch itself proves the mapping no longer
    /// refuses; cosine proves the reduced direction is the true one.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn proposal_joint_gradient_matches_binding_row_replay_with_channel_shared_alpha() {
        assert!(
            ny_gpu::wgpu_adapter_available(),
            "gpu-tests requires a usable WGPU adapter"
        );
        with_env_edits(|env| {
            env.set("NY_RESNET_WARMUP_GPU", "1");
            env.remove("NY_ROOT_ALPHA_GPU");
            env.remove("NY_BETA_GPU_PROBE");

            let (mut fix, _per_neuron_runtime) = joint_dispatchable_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_mins(2));
            let relus = ["relu0", "b1r1", "relu_out"];
            let alpha = mk_channel_alpha(&fix.graph, &fix.bounds, &relus, 0.45);
            let runtime = DagAlphaRuntimeState::new(
                alpha,
                None,
                relus.iter().map(|s| s.to_string()).collect(),
            );
            let steering =
                ny_gpu::GradientSteeringDevice::new_wgpu().expect("adapter probed available");
            let ctx = ctx_with_steering(&fix, None, Some(&steering));
            // Supplied local gradients at ALPHA width (the armed lane's
            // reality: backward/nonlinear.rs writes reduce_gradient output).
            let supplied: Vec<Array1<f32>> = relus
                .iter()
                .enumerate()
                .map(|(relu_index, name)| {
                    let len = runtime.graph().alpha(name).expect("alpha").len();
                    Array1::from_iter(
                        (0..len).map(|i| 1000.0 + relu_index as f32 * 10.0 + i as f32 * 0.01),
                    )
                })
                .collect();
            let binding_row = 0usize;
            let mut spec = vec![0.0f32; fix.output_dim];
            spec[binding_row] = 1.0;

            let result = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                )
                .expect("proposal lane resolves");
            assert_eq!(
                result.margin_dispatch,
                MarginGradientDispatch::JointDispatched {
                    source: MarginGradientJointSource::WgpuProposal
                },
                "channel-shared alpha must DISPATCH (the release A/B failure was \
                 local_fallback reason=mapping_mismatch on exactly this layout)"
            );

            // Independent oracle at the SAME channel-shared iterate.
            let mut grads_lower: Vec<Array1<f32>> = zero_local_grads(&fix);
            let mut grads_upper: Vec<Array1<f32>> = zero_local_grads(&fix);
            let (_, intermediate) = fix
                .graph
                .dag_alpha_backward_pass_with_intermediates(
                    &fix.input,
                    &fix.bounds,
                    &fix.exec_order,
                    fix.output_dim,
                    fix.input_dim,
                    runtime.relu_name_to_idx(),
                    runtime.graph(),
                    None,
                    &mut grads_lower,
                    &mut grads_upper,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("AnalyticChain intermediates fold");
            let replay = fix
                .graph
                .binding_row_true_alpha_grads(
                    &fix.input,
                    runtime.graph(),
                    &intermediate,
                    binding_row,
                )
                .expect("channel-shared binding-row replay");

            let mut dot = 0.0f64;
            let mut seam_sq = 0.0f64;
            let mut replay_sq = 0.0f64;
            let mut diff_sq = 0.0f64;
            for (index, name) in relus.iter().enumerate() {
                let seam = &result.gradients[index];
                let alpha_len = runtime.graph().alpha(name).expect("alpha").len();
                assert_eq!(
                    seam.len(),
                    alpha_len,
                    "seam output at '{name}' must be at ALPHA width C (host-side reduced)"
                );
                let oracle = replay
                    .grads
                    .get(*name)
                    .unwrap_or_else(|| panic!("replay gradient for ReLU '{name}'"));
                assert_eq!(seam.len(), oracle.len(), "width parity at '{name}'");
                for (s, o) in seam.iter().zip(oracle.iter()) {
                    dot += f64::from(*s) * f64::from(*o);
                    seam_sq += f64::from(*s) * f64::from(*s);
                    replay_sq += f64::from(*o) * f64::from(*o);
                    let diff = f64::from(*s) - f64::from(*o);
                    diff_sq += diff * diff;
                }
            }
            assert!(
                replay_sq > 0.0,
                "degenerate fixture: the replay oracle found no alpha sensitivity"
            );
            assert!(
                seam_sq > 0.0,
                "degenerate seam output: the reduced proposal gradient is identically zero"
            );
            let cosine = dot / (seam_sq.sqrt() * replay_sq.sqrt());
            let rel_l2 = (diff_sq / replay_sq.max(1e-30)).sqrt();
            eprintln!(
                "[proposal-vs-replay-ch] cosine={cosine:.6} rel_l2={rel_l2:.4e} |seam|={:.4e} |replay|={:.4e}",
                seam_sq.sqrt(),
                replay_sq.sqrt()
            );
            assert!(
                cosine > 0.98,
                "channel-reduced proposal gradient must point where the channel-summed \
                 binding-row replay points (cosine {cosine})"
            );
            assert!(
                rel_l2 < 5e-2,
                "channel-reduced proposal gradient must match the replay in magnitude too \
                 (rel_l2 {rel_l2})"
            );
        });
    }

    /// VERIFIER attack (deadline): an expired deadline reaching the
    /// channel-shared reduction branch must surface `DeadlineExpired` — never
    /// an `Ok` carrying a partial/mapped gradient, and never a silent
    /// `Ok(None)` mismatch demotion.
    #[test]
    fn verify_deadline_expiry_in_channel_reduction_is_typed_not_partial() {
        let fix = warmup_fixture();
        let alpha = mk_channel_alpha(&fix.graph, &fix.bounds, &CONV_FIXTURE_RELUS, 0.45);
        let ctx = ctx_of(&fix, None);
        let relu_names: Vec<String> = CONV_FIXTURE_RELUS.iter().map(|s| s.to_string()).collect();
        let supplied: Vec<Array1<f32>> = relu_names
            .iter()
            .map(|name| Array1::from_elem(alpha.alpha(name).expect("alpha").len(), 7.0))
            .collect();
        let fold: Vec<Vec<f32>> = fix
            .relu_nodes
            .iter()
            .map(|(_, width)| (0..*width).map(|i| i as f32).collect())
            .collect();
        // Sanity: with a live deadline this maps (the reduction fires).
        let ok = GraphNetwork::map_warmup_fold_gradients_with_deadline(
            &ctx,
            &alpha,
            &supplied,
            &relu_names,
            &fold,
            Instant::now() + Duration::from_mins(1),
        )
        .expect("no deadline error")
        .expect("channel layouts map");
        assert_eq!(ok.len(), supplied.len());
        // Expired deadline: typed error, not Ok(partial) and not Ok(None).
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("system uptime exceeds one millisecond");
        let result = GraphNetwork::map_warmup_fold_gradients_with_deadline(
            &ctx,
            &alpha,
            &supplied,
            &relu_names,
            &fold,
            expired,
        );
        assert!(
            matches!(result, Err(DeadlineJointAlphaFoldError::DeadlineExpired)),
            "expired deadline in the reduction path must be DeadlineExpired, got {result:?}"
        );
    }

    /// VERIFIER attack (mapper shape keying): a supplied/target width that
    /// merely DIVIDES the per-neuron gradient length — including the spatial
    /// count itself — must refuse in both twins; only the stored-alpha
    /// channel width reduces.
    #[test]
    fn verify_mapper_refuses_divisible_but_wrong_target_widths() {
        let fix = warmup_fixture();
        let alpha = mk_channel_alpha(&fix.graph, &fix.bounds, &CONV_FIXTURE_RELUS, 0.45);
        let ctx = ctx_of(&fix, None);
        let relu_names: Vec<String> = CONV_FIXTURE_RELUS.iter().map(|s| s.to_string()).collect();
        let supplied: Vec<Array1<f32>> = relu_names
            .iter()
            .map(|name| Array1::from_elem(alpha.alpha(name).expect("alpha").len(), 7.0))
            .collect();
        let fold: Vec<Vec<f32>> = fix
            .relu_nodes
            .iter()
            .map(|(_, width)| (0..*width).map(|i| i as f32).collect())
            .collect();
        // For each relu try target widths that divide C*H*W but are not the
        // stored alpha width: 1, spatial, and C*H*W/2 when even.
        let name0 = &relu_names[0];
        let shape = alpha.spatial_shape(name0).expect("geometry").to_vec();
        let spatial: usize = shape[1..].iter().product();
        let total = shape[0] * spatial;
        let mut bad_widths = vec![1usize, spatial];
        if total.is_multiple_of(2) && total / 2 != shape[0] {
            bad_widths.push(total / 2);
        }
        for bad in bad_widths {
            if bad == alpha.alpha(name0).expect("alpha").len() {
                continue;
            }
            let mut bad_supplied = supplied.clone();
            bad_supplied[0] = Array1::from_elem(bad, 7.0);
            assert!(
                GraphNetwork::map_warmup_fold_gradients(
                    &ctx,
                    &alpha,
                    &bad_supplied,
                    &relu_names,
                    &fold
                )
                .is_none(),
                "target width {bad} divides {total} but must refuse"
            );
            assert_eq!(
                GraphNetwork::map_warmup_fold_gradients_with_deadline(
                    &ctx,
                    &alpha,
                    &bad_supplied,
                    &relu_names,
                    &fold,
                    Instant::now() + Duration::from_mins(1),
                )
                .expect("no deadline error"),
                None,
                "deadline twin must refuse target width {bad} too"
            );
        }
    }

    // === #binding-row-replay production tier ===

    /// Env setup shared by the replay-tier tests: warmup gate on, every other
    /// α-policy gate cleared (I10 — the replay rides the existing armed
    /// margin gates only).
    fn replay_tier_env(env: &mut ny_test_utils::env::EnvEditor) {
        env.set("NY_RESNET_WARMUP_GPU", "1");
        env.remove("NY_ROOT_ALPHA_GPU");
        env.remove("NY_MULTIOBJ_JOINT_ALPHA");
        env.remove("NY_MULTIOBJ_JOINT_ALPHA_GPU");
        env.remove("NY_ROOT_ALPHA_TRUE");
        env.remove("NY_BETA_GPU_PROBE");
    }

    /// Direct no-deadline replay oracle at the same α iterate: one complete
    /// AnalyticChain intermediates fold + `binding_row_true_alpha_grads`.
    /// This pins the low-level math; finite production-seam tests separately
    /// assert that incomplete deadline-bounded captures refuse replay.
    fn direct_replay_oracle(
        fix: &WarmupFixture,
        runtime: &DagAlphaRuntimeState,
        binding_row: usize,
    ) -> crate::network::graph_alpha::binding_row_replay::BindingRowReplay {
        let mut grads_lower = zero_local_grads(fix);
        let mut grads_upper = zero_local_grads(fix);
        let (_, intermediate) = fix
            .graph
            .dag_alpha_backward_pass_with_intermediates(
                &fix.input,
                &fix.bounds,
                &fix.exec_order,
                fix.output_dim,
                fix.input_dim,
                runtime.relu_name_to_idx(),
                runtime.graph(),
                None,
                &mut grads_lower,
                &mut grads_upper,
                None,
                None,
                None,
                None,
            )
            .expect("oracle intermediates fold");
        fix.graph
            .binding_row_true_alpha_grads(&fix.input, runtime.graph(), &intermediate, binding_row)
            .expect("oracle binding-row replay")
    }

    /// Finite-authority Conv fixture: the bounded DAG backward exits through
    /// full-CROWN without publishing intermediate A-matrices. The replay must
    /// refuse that partial capture and preserve supplied-local fallback. The
    /// no-deadline direct oracle proves the replay math itself is available
    /// and distinct, so the refusal assertion is not vacuous.
    #[test]
    fn finite_conv_margin_lane_refuses_partial_replay_and_keeps_supplied_local() {
        with_env_edits(|env| {
            replay_tier_env(env);

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_mins(2));
            let runtime = runtime_for(&fix);
            let ctx = ctx_of(&fix, None);
            let supplied = sentinel_local_grads(&fix);
            let binding_row = 0usize;
            let mut spec = vec![0.0f32; fix.output_dim];
            spec[binding_row] = 1.0;

            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                ),
                &supplied,
                MarginGradientFallbackReason::ResidentUnavailable,
                MarginGradientFallbackSource::SuppliedLocal,
            );

            let oracle = direct_replay_oracle(&fix, &runtime, binding_row);
            for (index, (name, _)) in fix.relu_nodes.iter().enumerate() {
                let expected = oracle
                    .grads
                    .get(name)
                    .unwrap_or_else(|| panic!("oracle replay gradient for '{name}'"));
                assert_ne!(
                    grads_bits(std::slice::from_ref(expected)),
                    grads_bits(std::slice::from_ref(&supplied[index])),
                    "the complete no-deadline replay at '{name}' must differ from the finite \
                     seam's supplied-local fallback"
                );
            }
        });
    }

    /// The same finite-capture refusal with channel-shared alpha. The direct
    /// oracle still proves the replay produces correctly reduced alpha-width
    /// gradients when a complete no-deadline capture is explicitly requested.
    #[test]
    fn finite_conv_margin_lane_refuses_partial_channel_shared_replay() {
        with_env_edits(|env| {
            replay_tier_env(env);

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_mins(2));
            let alpha = mk_channel_alpha(&fix.graph, &fix.bounds, &CONV_FIXTURE_RELUS, 0.45);
            let runtime = DagAlphaRuntimeState::new(
                alpha,
                None,
                CONV_FIXTURE_RELUS.iter().map(|s| s.to_string()).collect(),
            );
            let ctx = ctx_of(&fix, None);
            // Supplied local gradients at ALPHA width (the armed lane's
            // reality: backward/nonlinear.rs writes reduce_gradient output).
            let supplied: Vec<Array1<f32>> = CONV_FIXTURE_RELUS
                .iter()
                .map(|name| {
                    Array1::from_elem(runtime.graph().alpha(name).expect("alpha").len(), 777.0)
                })
                .collect();
            let binding_row = 0usize;
            let mut spec = vec![0.0f32; fix.output_dim];
            spec[binding_row] = 1.0;

            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                ),
                &supplied,
                MarginGradientFallbackReason::ResidentUnavailable,
                MarginGradientFallbackSource::SuppliedLocal,
            );

            let oracle = direct_replay_oracle(&fix, &runtime, binding_row);
            for (index, name) in CONV_FIXTURE_RELUS.iter().enumerate() {
                let alpha_len = runtime.graph().alpha(name).expect("alpha").len();
                assert_eq!(
                    supplied[index].len(),
                    alpha_len,
                    "supplied fallback at '{name}' must be at ALPHA width C"
                );
                let expected = oracle
                    .grads
                    .get(*name)
                    .unwrap_or_else(|| panic!("oracle replay gradient for '{name}'"));
                assert_eq!(
                    expected.len(),
                    alpha_len,
                    "complete direct replay at '{name}' must reduce to ALPHA width C"
                );
                assert_ne!(
                    grads_bits(std::slice::from_ref(expected)),
                    grads_bits(std::slice::from_ref(&supplied[index])),
                    "the complete direct replay at '{name}' must differ from fallback"
                );
            }
        });
    }

    /// A typed proposal refusal still consults the replay tier, but the finite
    /// Conv capture cannot publish complete A-matrices. Preserve the original
    /// proposal's `joint_unavailable` reason and supplied-local gradients.
    #[test]
    fn finite_conv_proposal_refusal_survives_partial_replay_refusal() {
        with_env_edits(|env| {
            replay_tier_env(env);

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_mins(2));
            let runtime = runtime_for(&fix);
            let steering =
                ScriptedResnetGradEngine::proposal_only_with_behavior(ScriptedJointBehavior::Error);
            let ctx = ctx_with_steering(&fix, None, Some(&steering));
            let supplied = sentinel_local_grads(&fix);
            let binding_row = 0usize;
            let mut spec = vec![0.0f32; fix.output_dim];
            spec[binding_row] = 1.0;

            let result = fix.graph.try_gpu_resnet_warmup_gradients(
                &ctx,
                &fix.bounds,
                &runtime,
                &supplied,
                None,
                0,
                MarginGradientRequest::Binding(&spec),
            );
            assert_eq!(
                steering.joint_calls(),
                1,
                "the proposal accelerant must have been consulted first"
            );
            assert_local_margin_fallback(
                result,
                &supplied,
                MarginGradientFallbackReason::JointUnavailable,
                MarginGradientFallbackSource::SuppliedLocal,
            );
            let oracle = direct_replay_oracle(&fix, &runtime, binding_row);
            for (index, (name, _)) in fix.relu_nodes.iter().enumerate() {
                assert_ne!(
                    grads_bits(std::slice::from_ref(&oracle.grads[name.as_str()])),
                    grads_bits(std::slice::from_ref(&supplied[index])),
                    "the complete direct replay at '{name}' must remain non-vacuous"
                );
            }
        });
    }

    /// Gate 2b (tier order, dispatch side): when the proposal channel CAN
    /// dispatch it is preferred — measured accelerant order (this file's cost
    /// accounting: ~27s/iter CPU AnalyticChain intermediates capture vs 6 GPU
    /// dispatches in a 7s window, arbitration doc §6). The replay must NOT
    /// preempt a live proposal.
    #[test]
    fn proposal_accelerant_preferred_over_replay_when_it_dispatches() {
        with_env_edits(|env| {
            replay_tier_env(env);

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_mins(2));
            let runtime = runtime_for(&fix);
            let steering =
                ScriptedResnetGradEngine::proposal_only_with_behavior(ScriptedJointBehavior::Valid);
            let probe_ctx = ctx_of(&fix, None);
            let (_, fold_names, _, _) = fix
                .graph
                .warmup_segments(&probe_ctx, &fix.bounds, &runtime, "conv_out")
                .expect("proposal extraction");
            steering.set_joint_widths(
                fold_names
                    .iter()
                    .map(|name| {
                        fix.relu_nodes
                            .iter()
                            .find_map(|(runtime_name, width)| {
                                (runtime_name == name).then_some(*width)
                            })
                            .expect("fold ReLU belongs to the runtime")
                    })
                    .collect(),
            );
            let ctx = ctx_with_steering(&fix, None, Some(&steering));
            let supplied = sentinel_local_grads(&fix);
            let mut spec = vec![0.0f32; fix.output_dim];
            spec[0] = 1.0;

            let result = fix
                .graph
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                )
                .expect("armed lane resolves");
            assert_eq!(
                result.margin_dispatch,
                MarginGradientDispatch::JointDispatched {
                    source: MarginGradientJointSource::WgpuProposal
                },
                "a live proposal dispatch is the accelerant and must win the tier order"
            );
            assert_eq!(steering.joint_calls(), 1);
        });
    }

    /// Gate 2c (replay refusal → next tier, reason preserved): a
    /// multi-nonzero objective — the exact production margin-row shape
    /// `e_true − e_other` — is OUTSIDE the single-seed-row replay envelope
    /// (a general combination selects its own relaxation branches; the
    /// documented seed-row trap). With no proposal channel the armed lane
    /// must end on today's bounded local fallback, byte-identical; with a
    /// refusing proposal the PROPOSAL's typed reason must be preserved.
    #[test]
    fn replay_refuses_multi_row_objective_and_falls_through_with_reason_preserved() {
        with_env_edits(|env| {
            replay_tier_env(env);

            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_mins(2));
            let runtime = runtime_for(&fix);
            let supplied = sentinel_local_grads(&fix);
            let mut margin_row_spec = vec![0.0f32; fix.output_dim];
            margin_row_spec[0] = 1.0;
            margin_row_spec[1] = -1.0;

            // No proposal channel: replay refusal ⇒ the pre-existing moat.
            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx_of(&fix, None),
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&margin_row_spec),
                ),
                &supplied,
                MarginGradientFallbackReason::ResidentUnavailable,
                MarginGradientFallbackSource::SuppliedLocal,
            );

            // Refusing proposal + refusing replay: the proposal's typed
            // reason survives the replay tier.
            let steering =
                ScriptedResnetGradEngine::proposal_only_with_behavior(ScriptedJointBehavior::Error);
            assert_local_margin_fallback(
                fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx_with_steering(&fix, None, Some(&steering)),
                    &fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&margin_row_spec),
                ),
                &supplied,
                MarginGradientFallbackReason::JointUnavailable,
                MarginGradientFallbackSource::SuppliedLocal,
            );
            assert_eq!(steering.joint_calls(), 1);
        });
    }

    /// Gate 3 (bounded contract + disarmed trip-wire): without a loop
    /// deadline there is no bounded contract — the replay refuses exactly
    /// like the proposal channel and the armed lane keeps today's local
    /// fallback; with the child DISABLED the whole tier is unreachable
    /// (`None` ⇒ the legacy CPU path, byte-identical — the loop-level parity
    /// oracles `dag_alpha_loop_bit_identical_with_gate_on_and_off` and
    /// `conflicting_alpha_policies_keep_exact_child_disabled_loop` pin the
    /// end-to-end half).
    #[test]
    fn replay_requires_deadline_and_is_unreachable_when_disarmed() {
        with_env_edits(|env| {
            replay_tier_env(env);

            // Armed, one-hot, NO deadline ⇒ bounded local fallback.
            let no_deadline_fix = warmup_fixture();
            assert_eq!(no_deadline_fix.config.deadline, None);
            let runtime = runtime_for(&no_deadline_fix);
            let supplied = sentinel_local_grads(&no_deadline_fix);
            let mut spec = vec![0.0f32; no_deadline_fix.output_dim];
            spec[0] = 1.0;
            assert_local_margin_fallback(
                no_deadline_fix.graph.try_gpu_resnet_warmup_gradients(
                    &ctx_of(&no_deadline_fix, None),
                    &no_deadline_fix.bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                ),
                &supplied,
                MarginGradientFallbackReason::ResidentUnavailable,
                MarginGradientFallbackSource::SuppliedLocal,
            );

            // Disarmed on the same no-authority host WITH a deadline: the
            // attempt must stay `None` (legacy CPU path), proving the replay
            // tier is gated by the armed margin request alone (I10).
            let mut fix = warmup_fixture();
            fix.config.deadline = Some(Instant::now() + Duration::from_mins(2));
            let runtime = runtime_for(&fix);
            let supplied = sentinel_local_grads(&fix);
            assert!(
                fix.graph
                    .try_gpu_resnet_warmup_gradients(
                        &ctx_of(&fix, None),
                        &fix.bounds,
                        &runtime,
                        &supplied,
                        None,
                        0,
                        MarginGradientRequest::Disabled,
                    )
                    .is_none(),
                "child-disabled lane must keep the legacy CPU path byte-identical"
            );
        });
    }

    /// Loop-level hard-authority receipt: this finite Conv run cannot publish
    /// a complete intermediates capture, so it must finish via local fallback
    /// without incrementing the truthful accepted-replay counter.
    #[test]
    fn finite_conv_margin_loop_records_no_accepted_replay_dispatch() {
        let before = crate::alpha_gradient_steering::telemetry().replay_dispatches;
        let (_bounds, steering) = run_margin_loop_with_steering(None, 3);
        assert!(
            steering.is_none(),
            "this arm runs without a proposal channel"
        );
        let after = crate::alpha_gradient_steering::telemetry().replay_dispatches;
        assert_eq!(
            after, before,
            "a finite Conv loop must not report a replay after its bounded capture omitted \
             the A-matrices ({before} -> {after})"
        );
    }

    /// The seed-row objective filter (#binding-row-replay trap): exactly one
    /// positive finite nonzero maps; everything else refuses.
    #[test]
    fn single_positive_objective_row_filter() {
        assert_eq!(
            single_positive_objective_row(&[0.0, 2.5, 0.0]),
            Some((1, 2.5))
        );
        assert_eq!(single_positive_objective_row(&[1.0]), Some((0, 1.0)));
        assert_eq!(single_positive_objective_row(&[0.0, 0.0]), None);
        assert_eq!(single_positive_objective_row(&[]), None);
        assert_eq!(single_positive_objective_row(&[1.0, -1.0]), None);
        assert_eq!(single_positive_objective_row(&[-1.0, 0.0]), None);
        assert_eq!(single_positive_objective_row(&[f32::NAN, 0.0]), None);
        assert_eq!(single_positive_objective_row(&[f32::INFINITY]), None);
        // -0.0 == 0.0 in IEEE: treated as zero, not a (negative) row weight.
        assert_eq!(single_positive_objective_row(&[-0.0, 3.0]), Some((1, 3.0)));
    }

    /// ADVERSARIAL wrong-row attack (#binding-row-replay, the documented
    /// seed-row trap): under a published #margin-subset-alpha scope the
    /// capture seeds ONLY the k referenced rows, so `binding_row` is a
    /// COMPACT index — subset {5, 300} of 600 outputs puts output row 300 at
    /// seed row 1. The seam must call the replay with the COMPACT row and the
    /// result must be BITWISE the direct replay of seed row 1 (== output row
    /// 300) — and must DIFFER from seed row 0 (== output row 5), proving the
    /// mapping is real and not an accidental identity. An off-by-mapping here
    /// silently optimizes the wrong output row.
    #[test]
    fn subset_seeded_capture_maps_binding_row_through_published_indices() {
        use crate::layers::LinearLayer;
        use crate::network::core::GraphNode;
        use crate::network::graph_alpha::resnet_skeleton::test_support::lcg;
        use ndarray::Array2;

        with_env_edits(|env| {
            replay_tier_env(env);

            // MLP: input[8] → lin0(8→16) → relu0 → lin_out(16→600).
            // 600 ≥ MARGIN_SUBSET_MIN_OUTPUT_DIM so the published scope engages.
            let mut g = GraphNetwork::new();
            let mut rng = lcg(0x5EED_5EED_0001);
            let w0 = Array2::from_shape_fn((16, 8), |_| rng() * 0.6);
            let b0 = Array1::from_shape_fn(16, |_| rng() * 0.1);
            g.add_node(GraphNode::from_input(
                "lin0",
                crate::layers::Layer::Linear(LinearLayer::new(w0, Some(b0)).expect("lin0")),
            ));
            g.add_node(relu("relu0", "lin0"));
            let w1 = Array2::from_shape_fn((600, 16), |_| rng() * 0.4);
            let b1 = Array1::from_shape_fn(600, |_| rng() * 0.05);
            g.add_node(GraphNode::new(
                "lin_out",
                crate::layers::Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("lin_out")),
                vec!["relu0".to_string()],
            ));
            g.set_output("lin_out");

            let input = box_input(&[8], -1.0, 1.0);
            let bounds = g.collect_node_bounds(&input).expect("node bounds");
            let exec_order: Vec<String> = g.node_order.clone();
            let relu_width = bounds.get("lin0").expect("pre bounds").lower().len();
            assert_eq!(relu_width, 16);
            let relu_nodes = vec![("relu0".to_string(), relu_width)];
            let output_dim = bounds.get("lin_out").expect("out bounds").lower().len();
            assert_eq!(output_dim, 600);
            let input_dim = input.lower().len();
            let config = AlphaCrownConfig {
                deadline: Some(Instant::now() + Duration::from_mins(2)),
                ..Default::default()
            };

            let alpha = mk_alpha(&g, &bounds, &["relu0"], 0.35, 0.65);
            let runtime = DagAlphaRuntimeState::new(alpha, None, vec!["relu0".to_string()]);
            let ctx = DagAlphaLoopContext {
                input: &input,
                exec_order: &exec_order,
                output_dim,
                input_dim,
                config: &config,
                engine: None,
                alpha_steering: None,
                relu_nodes: &relu_nodes,
                has_bilinear: false,
                has_mul_binary: false,
            };
            let supplied: Vec<Array1<f32>> = vec![Array1::from_elem(relu_width, 555.0)];

            // Publish the margin subset {5, 300} for BOTH the seam call and
            // the oracle folds (same thread, same scope — the production
            // reality where the spec scope outlives the α loop).
            let _guard = crate::output_margin_seed::MarginOutputSeedGuard::publish(vec![5, 300]);
            assert_eq!(
                crate::output_margin_seed::margin_subset_indices(output_dim)
                    .as_deref()
                    .map(<[usize]>::to_vec),
                Some(vec![5, 300]),
                "the subset scope must be live"
            );

            // Armed one-hot objective on OUTPUT row 300 = subset position 1.
            let mut spec = vec![0.0f32; output_dim];
            spec[300] = 1.0;
            let result = g
                .try_gpu_resnet_warmup_gradients(
                    &ctx,
                    &bounds,
                    &runtime,
                    &supplied,
                    None,
                    0,
                    MarginGradientRequest::Binding(&spec),
                )
                .expect("armed lane resolves");
            assert_eq!(
                result.margin_dispatch,
                MarginGradientDispatch::JointDispatched {
                    source: MarginGradientJointSource::CpuReplay
                },
                "subset-scoped one-hot objective must dispatch the replay, not fall local"
            );

            // Direct oracles at the same iterate, under the same publication:
            // seed row 1 == output row 300 (right), seed row 0 == output row 5
            // (wrong).
            let mut grads_lower = vec![Array1::<f32>::zeros(relu_width)];
            let mut grads_upper = vec![Array1::<f32>::zeros(relu_width)];
            let (_bounds_out, intermediate) = g
                .dag_alpha_backward_pass_with_intermediates(
                    &input,
                    &bounds,
                    &exec_order,
                    output_dim,
                    input_dim,
                    runtime.relu_name_to_idx(),
                    runtime.graph(),
                    None,
                    &mut grads_lower,
                    &mut grads_upper,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("oracle intermediates fold");
            assert_eq!(
                intermediate.final_bounds.lower_a().nrows(),
                2,
                "the capture must have been subset-seeded (k=2 of 600 rows)"
            );
            let right = g
                .binding_row_true_alpha_grads(&input, runtime.graph(), &intermediate, 1)
                .expect("replay of seed row 1 (output row 300)");
            let wrong = g
                .binding_row_true_alpha_grads(&input, runtime.graph(), &intermediate, 0)
                .expect("replay of seed row 0 (output row 5)");
            let right_g = right.grads.get("relu0").expect("right grads");
            let wrong_g = wrong.grads.get("relu0").expect("wrong grads");
            assert!(
                right_g.iter().any(|v| *v != 0.0),
                "fixture must produce a nonzero true gradient for the attack to bite"
            );
            assert_ne!(
                grads_bits(std::slice::from_ref(right_g)),
                grads_bits(std::slice::from_ref(wrong_g)),
                "rows 5 and 300 must have distinguishable gradients or the test is vacuous"
            );
            assert_eq!(
                grads_bits(std::slice::from_ref(&result.gradients[0])),
                grads_bits(std::slice::from_ref(right_g)),
                "the seam must replay SEED row 1 (output row 300), bitwise"
            );
        });
    }
}
