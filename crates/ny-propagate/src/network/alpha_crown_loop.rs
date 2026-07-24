// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared alpha-CROWN optimization loop.
//!
//! This module contains the core optimization loop shared between
//! [`Network`] (sequential) and [`GraphNetwork`] (graph-based) alpha-CROWN
//! implementations. The loop is parameterized by [`AlphaCrownBackend`],
//! which abstracts the backward pass and gradient computation.
//!
//! Extracted to eliminate the ~80% code duplication between
//! `alpha_crown.rs` (Network) and `propagate_sequential.rs` (GraphNetwork).
//! Ensures structural consistency of NaN guards, early stopping, optimizer
//! updates, and best-bounds tracking across both paths. (#2835)

use crate::bounds::{
    AlphaCrownConfig, AlphaState, FacetBank, FacetBankSearchConfig, LinearBounds,
    LowerAffineCertificate, Optimizer, FACET_BANK_DEFAULT_DYADIC_BITS, FACET_BANK_MAX_PLANES,
};
use crate::network::graph_alpha::invprop_backward::take_best_bounds;
use crate::network::graph_alpha::propagate_helpers::{
    bounds_infeasible, clamp_inverted_best_bounds, update_elementwise_best_bounds,
};
use ndarray::{Array1, ArrayD};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::{any::Any, mem::size_of, time::Instant};
use tracing::{debug, info, warn};

/// Gradient pair: (lower_path_gradients, upper_path_gradients) per ReLU layer.
pub(crate) type DualGradients = (Vec<Array1<f32>>, Vec<Array1<f32>>);

/// Result of a single backward pass iteration.
pub(crate) struct BackwardIterationResult {
    /// Linear bounds from the backward pass.
    pub(crate) linear_bounds: LinearBounds,
    /// Gradients for each ReLU layer for lower alpha path (one `Array1` per ReLU).
    pub(crate) gradients: Vec<Array1<f32>>,
    /// Gradients for each ReLU layer for upper alpha path (#3393).
    pub(crate) gradients_upper: Vec<Array1<f32>>,
    /// Optional bounds computed without output constraints (for INVPROP best-of).
    pub(crate) bounds_without_oc: Option<LinearBounds>,
}

/// Opaque reference-bounds refresh payload produced and consumed by backends.
///
/// The shared loop only transports the candidate from `post_bounds_update` to
/// `apply_reference_refresh`; backend implementations own the concrete type.
pub(crate) struct ReferenceBoundsCandidate {
    pub(crate) data: Box<dyn Any>,
}

/// Backend trait for alpha-CROWN backward pass and gradient computation.
///
/// Implementors provide the network-specific backward pass (sequential layers
/// vs graph nodes) and gradient computation (SPSA/finite-diff/analytic).
/// The shared loop handles everything else: best-bounds tracking, early stopping,
/// non-finite gradient guards, optimizer updates, and lr decay.
///
/// Optional hooks (with default no-ops) extend the loop for GraphNetwork features:
/// reference bounds refresh, pilot iteration check, extended alpha types, and
/// debug telemetry. Sequential backends use defaults and are unaffected. (#1948)
pub(crate) trait AlphaCrownBackend {
    /// Run one backward pass iteration with current alpha values.
    ///
    /// Returns `Ok(Some(result))` for a successful backward pass,
    /// `Ok(None)` to signal a fallback to CROWN (e.g., unsupported op),
    /// or `Err(e)` for a real error.
    fn backward_iteration(
        &self,
        alpha_state: &AlphaState,
        input: &BoundedTensor,
        iter: usize,
        invprop_enabled: bool,
        need_grad: bool,
    ) -> Result<Option<BackwardIterationResult>>;

    /// Compute gradients for the current iteration.
    ///
    /// Called after the backward pass succeeds. The `gradients` and `gradients_upper`
    /// parameters contain the analytic gradients from the backward pass for lower
    /// and upper alpha paths respectively (#3393). Returns (lower_grads, upper_grads).
    /// For numerical methods (SPSA, FiniteDiff), the same gradient is used for both
    /// paths since they are perturbed jointly.
    fn compute_gradients(
        &self,
        config: &AlphaCrownConfig,
        alpha_state: &mut AlphaState,
        input: &BoundedTensor,
        gradients: &[Array1<f32>],
        gradients_upper: &[Array1<f32>],
        iter: usize,
    ) -> Result<DualGradients>;

    /// Fall back to CROWN bounds (used when alpha-CROWN cannot proceed).
    fn crown_fallback(&self, input: &BoundedTensor) -> Result<BoundedTensor>;

    /// Label for debug/info log messages (e.g., "α-CROWN", "GraphNetwork α-CROWN").
    fn log_label(&self) -> &str;

    // --- Optional hooks for GraphNetwork features (#1948) ---
    // Default implementations are no-ops so existing backends (SequentialAlphaCrownBackend)
    // continue working without changes.

    /// Collect an optional reference-bounds refresh candidate for this iteration.
    ///
    /// GraphNetwork backends use this to snapshot tighter activation-input bounds
    /// before gradient computation mutates optimization state, mirroring the DAG
    /// alpha-CROWN loop and alpha-beta-CROWN `best_intermediate_bounds` refresh
    /// pattern in `optimized_bounds.py:338-367,500-615`.
    ///
    /// Default: no refresh candidate.
    fn post_bounds_update(
        &mut self,
        _iter: usize,
        _improved_output: bool,
    ) -> Result<Option<ReferenceBoundsCandidate>> {
        Ok(None)
    }

    /// Apply a previously collected reference-bounds refresh candidate.
    ///
    /// The shared loop calls this after alpha updates and INVPROP ny clipping,
    /// matching the original DAG alpha-CROWN ordering.
    ///
    /// Default: no-op.
    fn apply_reference_refresh(
        &mut self,
        candidate: ReferenceBoundsCandidate,
        _iter: usize,
    ) -> Result<()> {
        drop(candidate.data);
        Ok(())
    }

    /// Called after first alpha update (iter == 1). Return `true` to abort
    /// optimization early (pilot iteration didn't help).
    ///
    /// GraphNetwork DAG backends use this for adaptive skip: if the first
    /// alpha iteration didn't improve bounds beyond CROWN, the remaining
    /// iterations are unlikely to help and the budget is better spent elsewhere.
    ///
    /// Default: `false` (never abort early).
    fn pilot_check(
        &self,
        _config: &AlphaCrownConfig,
        _best_lower_sum: f32,
        _crown_bounds: &BoundedTensor,
    ) -> bool {
        false
    }

    /// Update extended alpha parameters beyond ReLU (bilinear, MulBinary,
    /// S-shaped activations). Called after the standard ReLU alpha update.
    ///
    /// GraphNetwork DAG backends use this for Adam/SGD updates to bilinear,
    /// MulBinary, and S-shaped (Sigmoid/Tanh) alpha parameters.
    ///
    /// Default: no-op.
    fn update_extended_alphas(
        &mut self,
        _config: &AlphaCrownConfig,
        _lr: f32,
        _iter: usize,
        _total_gradient_skips: &mut usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Per-iteration debug telemetry. Called at the end of each iteration
    /// when `iter % 5 == 0` and DEBUG tracing is enabled.
    ///
    /// GraphNetwork DAG backends use this to log alpha/velocity statistics.
    ///
    /// Default: no-op.
    fn log_iteration_telemetry(&self, _iter: usize) {}
}

/// Sum only finite elements, treating non-finite (NaN, ±Inf) as zero.
///
/// This prevents a single `-Inf` lower bound from poisoning the scalar early-stopping
/// metric: `-Inf + finite = -Inf`, making `(-Inf) - (-Inf) = NaN` and disabling
/// convergence detection entirely. With this helper, infinite elements are neutral
/// and early stopping tracks improvement of the finite dimensions only. (#2857)
pub(crate) fn finite_lower_sum(arr: &ArrayD<f32>) -> f32 {
    arr.iter().filter(|v| v.is_finite()).copied().sum::<f32>()
}

/// Run the shared alpha-CROWN optimization loop.
///
/// This function encapsulates the entire optimization procedure:
/// 1. Initialize best bounds from CROWN
/// 2. For each iteration: backward pass → concretize → update best → compute gradients → update alpha
/// 3. Return element-wise best bounds or CROWN fallback
///
/// The `invprop_enabled` flag controls INVPROP-specific behavior (infeasibility
/// detection, ny clipping).
/// Max iterations that run the (one-extra-backward) INVPROP gamma ascent probe.
/// Gammas are few and converge fast, so a small cap bounds the throughput cost of
/// on-by-default INVPROP (see the throughput guard at the call site).
const INVPROP_ASCENT_MAX_ITERS: usize = 5;

/// Dark B1 gate: evaluate the planned terminal alpha iterate as a certified
/// bound-only pass, then persist exactly the state that produced that bound.
///
/// Only the exact value `"1"` enables the experiment. Unset, malformed, and
/// explicit-zero values preserve the legacy gradient/update path.
fn parse_final_alpha_bound_only(raw: Option<&str>) -> bool {
    raw == Some("1")
}

pub(crate) fn final_alpha_bound_only_enabled() -> bool {
    let raw = std::env::var("NY_ALPHA_FINAL_BOUND_ONLY").ok();
    parse_final_alpha_bound_only(raw.as_deref())
}

/// Whether an optimization iteration can feed another evaluated alpha bound.
///
/// Kept as a pure helper so the shared and DAG loops cannot drift on zero/one/
/// many-iteration scheduling.
pub(crate) fn alpha_iteration_needs_gradient(
    iter: usize,
    iterations: usize,
    final_bound_only: bool,
) -> bool {
    !(final_bound_only && iterations > 0 && iter == iterations - 1)
}

// FacetBank is an experimental, opt-in tightening pass.  Keeping this gate here
// makes the default alpha-CROWN path byte-for-byte unchanged while the VNN-COMP
// presets are being measured.  The collector is deliberately bounded: retained
// coefficient centers plus worst-case error-vector payload defaults to 64 MiB.
const ALPHA_FACET_BANK_DEFAULT_PLANES: usize = 4;
const ALPHA_FACET_BANK_DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;
const ALPHA_FACET_BANK_REFINEMENT_ROUNDS: usize = 2;

#[derive(Clone, Copy, Debug)]
struct AlphaFacetBankSettings {
    enabled: bool,
    max_planes: usize,
    max_bytes: usize,
}

impl AlphaFacetBankSettings {
    fn from_env() -> Self {
        let switch = |name: &str| {
            std::env::var(name).ok().and_then(|raw| {
                match raw.trim().to_ascii_lowercase().as_str() {
                    "1" | "true" | "on" => Some(true),
                    "0" | "false" | "off" => Some(false),
                    _ => None,
                }
            })
        };
        // The master switch enables the complete Hydra experiment, while an
        // explicit component value remains an override for factorial A/Bs.
        let enabled =
            switch("NY_FACET_BANK").unwrap_or_else(|| switch("NY_HYDRA_CROWN").unwrap_or(false));
        let max_planes = std::env::var("NY_FACET_BANK_PLANES")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(ALPHA_FACET_BANK_DEFAULT_PLANES)
            .clamp(2, FACET_BANK_MAX_PLANES);
        let max_bytes = std::env::var("NY_FACET_BANK_MAX_BYTES")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(ALPHA_FACET_BANK_DEFAULT_MAX_BYTES);
        Self {
            enabled,
            max_planes,
            max_bytes,
        }
    }
}

/// A bounded trajectory cache for input-relative lower affine forms.
///
/// For each output row, the plain CROWN concrete lower bound is retained as a
/// zero-coefficient constant anchor. The next `max_planes - 2` alpha iterates
/// provide trajectory diversity, and the final slot is continuously replaced
/// with the newest iterate. Thus an early-stopped run still contains its CROWN
/// baseline, early trajectory, and actual terminal plane without unbounded
/// memory growth.
struct AlphaFacetBankCollector {
    rows: Vec<Vec<LowerAffineCertificate>>,
    input_dim: usize,
    max_planes: usize,
}

impl AlphaFacetBankCollector {
    fn new(output_dim: usize, input_dim: usize, settings: AlphaFacetBankSettings) -> Option<Self> {
        if !settings.enabled || output_dim == 0 || input_dim == 0 {
            return None;
        }

        // Budget for both a center and a possible symmetric error per retained
        // coefficient. Bias/vector overhead is negligible but included as one
        // extra coefficient per row. All arithmetic fails closed on overflow.
        let required_bytes = output_dim
            .checked_mul(input_dim.checked_add(1)?)?
            .checked_mul(settings.max_planes)?
            .checked_mul(2 * size_of::<f32>())?;
        if required_bytes > settings.max_bytes {
            debug!(
                required_bytes,
                max_bytes = settings.max_bytes,
                "alpha-CROWN FacetBank disabled by memory cap"
            );
            return None;
        }

        Some(Self {
            rows: (0..output_dim)
                .map(|_| Vec::with_capacity(settings.max_planes))
                .collect(),
            input_dim,
            max_planes: settings.max_planes,
        })
    }

    /// Retain the initial concrete CROWN lower bounds as constant affine forms.
    /// A scalar `c <= f(x)` over the whole input domain is exactly the valid
    /// zero-coefficient certificate `0*x + c <= f(x)` and can therefore be
    /// mixed with every later input-relative alpha certificate.
    fn capture_constant_lower(&mut self, lower: &ArrayD<f32>) {
        if lower.len() != self.rows.len() {
            debug!("alpha-CROWN FacetBank skipped a shape-mismatched CROWN anchor");
            return;
        }
        for (&bias, retained) in lower.iter().zip(&mut self.rows) {
            if !bias.is_finite() {
                continue;
            }
            if let Ok(certificate) = LowerAffineCertificate::new(vec![0.0; self.input_dim], bias) {
                retained.push(certificate);
            }
        }
    }

    /// Retain all valid lower rows from one sound CROWN backward result.
    /// Invalid rows are ignored because this optional tightening pass must never
    /// turn a valid baseline verification into an error.
    fn capture(&mut self, bounds: &LinearBounds) {
        if bounds.num_outputs() != self.rows.len() || bounds.num_inputs() != self.input_dim {
            debug!(
                expected_outputs = self.rows.len(),
                expected_inputs = self.input_dim,
                actual_outputs = bounds.num_outputs(),
                actual_inputs = bounds.num_inputs(),
                "alpha-CROWN FacetBank skipped a shape-mismatched iterate"
            );
            return;
        }

        let lower_errors = bounds.lower_a_err();
        let mut skipped = 0usize;
        for (row_index, retained) in self.rows.iter_mut().enumerate() {
            let coefficients = bounds.lower_a().row(row_index).to_vec();
            let coefficient_errors = lower_errors.map(|errors| errors.row(row_index).to_vec());
            let certificate = LowerAffineCertificate::from_parts(
                coefficients,
                bounds.lower_b()[row_index],
                coefficient_errors,
                0.0,
            );
            let Ok(certificate) = certificate else {
                skipped += 1;
                continue;
            };
            if retained.len() < self.max_planes {
                retained.push(certificate);
            } else {
                retained[self.max_planes - 1] = certificate;
            }
        }
        if skipped > 0 {
            debug!(skipped, "alpha-CROWN FacetBank skipped invalid affine rows");
        }
    }

    /// Certify retained simplex mixtures and monotonically merge them into the
    /// existing lower bounds. Every error is local to a row and becomes a
    /// fail-safe no-op.
    fn merge_into(
        self,
        input: &BoundedTensor,
        best_lower: &mut ArrayD<f32>,
        best_upper: &ArrayD<f32>,
        label: &str,
        deadline: Option<Instant>,
    ) {
        if best_lower.len() != self.rows.len() || best_upper.len() != self.rows.len() {
            debug!("{label}: FacetBank skipped final shape-mismatched merge");
            return;
        }
        let search = FacetBankSearchConfig {
            dyadic_bits: FACET_BANK_DEFAULT_DYADIC_BITS,
            refinement_rounds: ALPHA_FACET_BANK_REFINEMENT_ROUNDS,
        };
        let mut improved_rows = 0usize;
        let mut max_gain = 0.0f32;
        for ((certificates, lower), &upper) in self
            .rows
            .into_iter()
            .zip(best_lower.iter_mut())
            .zip(best_upper.iter())
        {
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                debug!("{label}: FacetBank stopped at the alpha-CROWN deadline");
                break;
            }
            if certificates.len() < 2 || lower.is_nan() || upper.is_nan() {
                continue;
            }
            let Ok(bank) = FacetBank::from_certificates(certificates, search) else {
                continue;
            };
            let Ok(certified) = bank.certify(input) else {
                continue;
            };
            let candidate = certified.lower_bound;
            // The explicit `candidate <= upper` gate avoids manufacturing an
            // inverted interval if an upstream numerical anomaly escaped its
            // own firewall. A sound finite candidate otherwise only tightens.
            if candidate.is_finite() && candidate > *lower && candidate <= upper {
                max_gain = max_gain.max(candidate - *lower);
                *lower = candidate;
                improved_rows += 1;
            }
        }
        if improved_rows > 0 {
            info!(
                "{label}: FacetBank convexification tightened {improved_rows} output rows \
                 (max gain {max_gain:.6})"
            );
        }
    }
}

/// One projected-ascent step on the INVPROP output-seed duals (Stage 2).
///
/// SOUNDNESS: this only mutates the gamma multipliers used by the *next*
/// backward's seed fold. Every reported bound still goes through the sound,
/// directed-rounding, sign-aware augment + `concretize_sound`, and the
/// best-bounds merge keeps the tightest SOUND iterate — so a wrong or suboptimal
/// gamma can only fail to improve, never inflate a bound. The gradient therefore
/// need not be exact or certified; a cheap **deterministic** one-sided SPSA
/// estimate (reproducible verdicts) is sufficient to steer tightness.
///
/// The extra backward here is a pure gradient probe; its bounds are discarded and
/// never feed the verdict.
fn invprop_seed_gamma_ascent_step<B: AlphaCrownBackend>(
    backend: &B,
    config: &AlphaCrownConfig,
    alpha_state: &mut AlphaState,
    input: &BoundedTensor,
    iter: usize,
    base_obj: f32,
) -> Result<()> {
    if !base_obj.is_finite() {
        return Ok(());
    }
    // Snapshot the current seed duals.
    let seed = match alpha_state
        .invprop_state
        .as_ref()
        .and_then(|s| s.layer_gammas(crate::invprop::INVPROP_OUTPUT_SEED))
    {
        Some(g) if g.active && !g.gammas.is_empty() => g.gammas.clone(),
        _ => return Ok(()),
    };

    let lr = if config.invprop.gamma_lr > 0.0 {
        config.invprop.gamma_lr
    } else {
        0.5
    };
    let delta = 0.1f32; // SPSA probe magnitude

    // Deterministic +/-1 perturbation sign per entry (reproducible across runs).
    let sign = |idx: usize| -> f32 {
        let h = (iter.wrapping_mul(2_654_435_761) ^ idx.wrapping_mul(40_503)) & 1;
        if h == 0 {
            1.0
        } else {
            -1.0
        }
    };

    let restore = |alpha_state: &mut AlphaState, g: &ndarray::Array3<f32>| {
        if let Some(gm) = alpha_state
            .invprop_state
            .as_mut()
            .and_then(|s| s.layer_gammas_mut(crate::invprop::INVPROP_OUTPUT_SEED))
        {
            gm.gammas.assign(g);
        }
    };

    // Probe: gamma + delta*sign (projected >= 0).
    let mut perturbed = seed.clone();
    for (idx, v) in perturbed.iter_mut().enumerate() {
        *v = (*v + delta * sign(idx)).max(0.0);
    }
    restore(alpha_state, &perturbed);

    let obj_plus = match backend.backward_iteration(alpha_state, input, iter, true, true)? {
        Some(bp) => {
            let mut cb = bp.linear_bounds.concretize_sound(input);
            if let Some(no_oc) = bp.bounds_without_oc {
                cb = take_best_bounds(&cb, &no_oc.concretize_sound(input));
            }
            finite_lower_sum(cb.lower())
        }
        None => {
            restore(alpha_state, &seed);
            return Ok(());
        }
    };

    // One-sided SPSA gradient of the maximize-lower-sum objective.
    let scale = (obj_plus - base_obj) / delta;
    if !scale.is_finite() {
        restore(alpha_state, &seed);
        return Ok(());
    }

    // Ascent from the ORIGINAL duals; projection >= 0 (clip_gammas re-projects too).
    let mut updated = seed;
    for (idx, v) in updated.iter_mut().enumerate() {
        *v = (*v + lr * scale * sign(idx)).max(0.0);
    }
    restore(alpha_state, &updated);
    Ok(())
}

pub(crate) fn alpha_crown_optimize<B: AlphaCrownBackend>(
    backend: &mut B,
    config: &AlphaCrownConfig,
    alpha_state: &mut AlphaState,
    input: &BoundedTensor,
    invprop_enabled: bool,
) -> Result<BoundedTensor> {
    alpha_crown_optimize_impl(
        backend,
        config,
        alpha_state,
        input,
        invprop_enabled,
        AlphaFacetBankSettings::from_env(),
        final_alpha_bound_only_enabled(),
    )
}

fn alpha_crown_optimize_impl<B: AlphaCrownBackend>(
    backend: &mut B,
    config: &AlphaCrownConfig,
    alpha_state: &mut AlphaState,
    input: &BoundedTensor,
    invprop_enabled: bool,
    facet_settings: AlphaFacetBankSettings,
    final_bound_only: bool,
) -> Result<BoundedTensor> {
    // Own the label to avoid holding an immutable borrow on `backend` across
    // mutable hook calls.
    let label: String = backend.log_label().to_string();
    let label = label.as_str();

    // Initialize best bounds from CROWN
    let crown_bounds = backend.crown_fallback(input)?;
    let mut best_lower: ArrayD<f32> = crown_bounds.lower().clone();
    let mut best_upper: ArrayD<f32> = crown_bounds.upper().clone();
    // Use finite-only sum to prevent -Inf poisoning the early-stopping metric (#2857).
    // Prior layout-agnostic fix: #1939.
    let mut best_lower_sum: f32 = finite_lower_sum(crown_bounds.lower());
    let mut prev_best_lower_sum = best_lower_sum;
    let mut no_improve_iters = 0usize;
    let mut lr = config.learning_rate;
    let mut infeasible_bounds: Option<BoundedTensor> = None;
    let mut total_gradient_skips: usize = 0;
    let mut facet_collector =
        AlphaFacetBankCollector::new(best_lower.len(), input.len(), facet_settings);
    if let Some(collector) = facet_collector.as_mut() {
        collector.capture_constant_lower(crown_bounds.lower());
    }

    for iter in 0..config.iterations {
        // Deadline check (#2698): bail early if verification timeout budget
        // is exhausted. Return current best bounds instead of running all iterations.
        if config.past_deadline() {
            info!(
                "{label}: deadline exceeded at iteration {}/{}, returning best bounds",
                iter, config.iterations
            );
            break;
        }

        // Compute this before dispatch: analytic backends can avoid producing
        // gradients in the same pass that computes the terminal certified bound.
        let need_grad = alpha_iteration_needs_gradient(iter, config.iterations, final_bound_only);

        // Run backward pass.
        let bp_result = match backend.backward_iteration(
            alpha_state,
            input,
            iter,
            invprop_enabled,
            need_grad,
        )? {
            Some(result) => result,
            None => return backend.crown_fallback(input),
        };

        // Coefficient retention is optional and can copy a large matrix.  If the
        // backward itself crossed the deadline, preserve its scalar result but do
        // not spend post-deadline time materializing another trajectory facet.
        if !config.past_deadline() {
            if let Some(collector) = facet_collector.as_mut() {
                collector.capture(&bp_result.linear_bounds);
            }
        }

        // Concretize to get actual bounds
        let mut concrete_bounds = bp_result.linear_bounds.concretize_sound(input);
        if let Some(bounds_no_oc) = bp_result.bounds_without_oc {
            let no_oc_bounds = bounds_no_oc.concretize_sound(input);
            concrete_bounds = take_best_bounds(&concrete_bounds, &no_oc_bounds);
        }

        // INVPROP infeasibility check
        if let Some(ref mut state) = alpha_state.invprop_state {
            if bounds_infeasible(&concrete_bounds) {
                state.mark_infeasible(0)?;
                state.apply_infeasible_mask(&mut concrete_bounds);
                infeasible_bounds = Some(concrete_bounds);
                break;
            }
        }

        // Update element-wise best bounds (layout-agnostic and shape-agnostic).
        // Skip during warmup window to avoid locking in noisy early-iteration bounds.
        // Matches α,β-CROWN's start_save_best (optimized_bounds.py:785-797).
        // `force` is true on the last iteration to ensure we always capture final bounds.
        let is_last_iter = iter == config.iterations - 1;
        if config.should_save_best(iter, is_last_iter) {
            update_elementwise_best_bounds(
                &mut best_lower,
                &mut best_upper,
                &concrete_bounds,
                iter,
            )?;
        }

        // Finite-only sum for early stopping (#2857). Layout-agnostic (#1939).
        let lower_sum: f32 = finite_lower_sum(concrete_bounds.lower());

        // NaN detection: if any bound element is NaN, the backward pass produced
        // garbage. Break early to avoid wasting remaining iterations — the
        // post-loop has_nan check will fall back to CROWN. (#2597)
        if concrete_bounds.lower().iter().any(|v| v.is_nan())
            || concrete_bounds.upper().iter().any(|v| v.is_nan())
        {
            warn!("{label}: NaN in bounds at iteration {iter}, aborting optimization (#2597)");
            break;
        }

        // Track best lower_sum for early stopping
        let improved_output = lower_sum > best_lower_sum;
        if improved_output {
            best_lower_sum = lower_sum;
        }

        // Early stopping check (compare best improvement since last iteration).
        let best_improvement = best_lower_sum - prev_best_lower_sum;
        if best_improvement < config.tolerance {
            no_improve_iters += 1;
        } else {
            no_improve_iters = 0;
        }
        if iter > 0 && no_improve_iters >= config.early_stop_patience {
            // Force-save before early exit to avoid losing optimization progress
            // when patience is exhausted during the warmup window.
            // Reference: optimized_bounds.py:794 (patience == early_stop_patience).
            if !config.should_save_best(iter, false) {
                update_elementwise_best_bounds(
                    &mut best_lower,
                    &mut best_upper,
                    &concrete_bounds,
                    iter,
                )?;
            }
            debug!(
                "{label}: Converged at iteration {} (best improvement < {} for {} iters)",
                iter, config.tolerance, no_improve_iters
            );
            break;
        }

        // Pilot iteration check (#1948): after the first alpha-updated iteration
        // (iter == 1), verify optimization helped before spending another update.
        // The DAG alpha-CROWN path performs this check before gradient computation
        // and parameter mutation for iteration 1.
        if iter == 1 && backend.pilot_check(config, best_lower_sum, &crown_bounds) {
            debug!("{label}: pilot check aborted at iteration 1 (#1948)");
            if let Some(collector) = facet_collector.take() {
                collector.merge_into(input, &mut best_lower, &best_upper, label, config.deadline);
            }
            clamp_inverted_best_bounds(&mut best_lower, &mut best_upper, label);
            return BoundedTensor::new_allow_infinite(best_lower, best_upper).map_err(|e| {
                NyError::InternalError(format!("{label}: pilot abort bounds invalid: {e}"))
            });
        }

        // The terminal bound has now passed every validity, infeasibility,
        // best-bound, patience, and pilot check. Nothing below can influence a
        // later evaluated bound, so preserve the exact state that produced this
        // one and skip all gradient/probe/reference/optimizer mutations.
        if !need_grad {
            debug!(
                method = ?config.gradient_method,
                iter,
                skipped_gradient_dispatches = 1usize,
                skipped_state_updates = 1usize,
                "{label}: NY_ALPHA_FINAL_BOUND_ONLY terminal pass"
            );
            break;
        }

        let refresh_candidate = backend.post_bounds_update(iter, improved_output)?;

        // Compute gradients via backend (returns separate lower/upper gradients, #3393)
        let (numerical_gradients, numerical_gradients_upper) = backend.compute_gradients(
            config,
            alpha_state,
            input,
            &bp_result.gradients,
            &bp_result.gradients_upper,
            iter,
        )?;

        // Debug output for first iteration (only when RUST_LOG=debug)
        if iter == 0 {
            for (relu_idx, grad) in numerical_gradients.iter().enumerate() {
                let grad_norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
                debug!(
                    "{label} iter 0: ReLU layer {} gradient L2 norm={:.6}",
                    relu_idx, grad_norm,
                );
            }
        }

        // Update alpha using gradient (gradient ascent to maximize lower bound).
        // For gradient ascent, negate the gradient (we want to maximize lower bound).
        // Update both lower and upper alpha paths independently (#3393).
        let adam_params = config.adam_params(lr, iter + 1);
        for (relu_idx, grad) in numerical_gradients.iter().enumerate() {
            // Guard: reject non-finite gradients before optimizer update.
            // Without this gate, NaN/Inf gradients from numerical instability
            // in the backward pass or chain-rule computation silently enter
            // the optimizer state (m/v for Adam, velocity for SGD), where the
            // downstream NaN sanitization in alpha.rs masks the root cause
            // by resetting alpha to 0.5. Skipping the update preserves the
            // current alpha (a better heuristic than silent reset). (#2809, #2835)
            if grad.iter().any(|v| !v.is_finite()) {
                warn!(
                    "{label} iter {}: skipping ReLU {} gradient update — non-finite values detected (#2835)",
                    iter, relu_idx
                );
                total_gradient_skips += 1;
                continue;
            }
            let neg_grad = grad.mapv(|v| -v);
            match config.optimizer {
                Optimizer::Adam => {
                    alpha_state.update_adam(relu_idx, &neg_grad, &adam_params);
                }
                Optimizer::Sgd => {
                    let momentum = if config.use_momentum {
                        config.momentum
                    } else {
                        0.0
                    };
                    alpha_state.update(relu_idx, &neg_grad, lr, momentum);
                }
            }

            // Update upper alpha path (#3393).
            if let Some(grad_upper) = numerical_gradients_upper.get(relu_idx) {
                if grad_upper.iter().any(|v| !v.is_finite()) {
                    continue;
                }
                let neg_grad_upper = grad_upper.mapv(|v| -v);
                match config.optimizer {
                    Optimizer::Adam => {
                        alpha_state.update_adam_upper(relu_idx, &neg_grad_upper, &adam_params);
                    }
                    Optimizer::Sgd => {
                        let momentum = if config.use_momentum {
                            config.momentum
                        } else {
                            0.0
                        };
                        alpha_state.update_upper(relu_idx, &neg_grad_upper, lr, momentum);
                    }
                }
            }
        }

        // Extended alpha updates (#1948): bilinear, MulBinary, S-shaped activations.
        // Called after the standard ReLU alpha update so all gradient-based updates
        // happen in sequence before reference bounds refresh.
        backend.update_extended_alphas(config, lr, iter, &mut total_gradient_skips)?;

        // INVPROP: projected gamma ascent on the output-seed duals (Stage 2).
        // Gated on optimize_gammas (default off => byte-identical baseline). Runs
        // BEFORE clip so the >= 0 projection is applied uniformly. Soundness is
        // independent of the gamma value (see helper docs), so a cheap SPSA step
        // suffices; the verdict always comes from the sound best-bounds merge.
        //
        // THROUGHPUT GUARD: each step costs one extra backward, so it is capped to
        // the first few iterations (gammas are few and converge fast) and skipped
        // once the deadline is near. This bounds the overhead so on-by-default can
        // only help or no-op, never regress a budget into a timeout.
        if invprop_enabled
            && config.invprop.optimize_gammas
            && iter < INVPROP_ASCENT_MAX_ITERS
            && !config.past_deadline()
        {
            invprop_seed_gamma_ascent_step(&*backend, config, alpha_state, input, iter, lower_sum)?;
        }

        // Clip gammas to enforce non-negativity (INVPROP constraint)
        if invprop_enabled {
            alpha_state.clip_gammas();
        }

        // Reference bounds refresh (#1948): apply the candidate collected before
        // gradient computation after alpha updates and ny clipping.
        if let Some(candidate) = refresh_candidate {
            backend.apply_reference_refresh(candidate, iter)?;
        }

        // Learning rate decay
        lr *= config.lr_decay;

        if iter % 5 == 0 {
            debug!(
                "{label} iter {}: lower_sum = {:.6}, lr = {:.6}",
                iter, lower_sum, lr,
            );
            // Extended telemetry (#1948)
            if tracing::enabled!(tracing::Level::DEBUG) {
                backend.log_iteration_telemetry(iter);
            }
        }

        prev_best_lower_sum = best_lower_sum;
    }
    // Gradient skip summary (#2981 Slice 5).
    if total_gradient_skips > 0 {
        warn!(
            "{label}: skipped {total_gradient_skips}/{} gradient updates (non-finite)",
            config.iterations * alpha_state.alphas.len()
        );
    }

    if let Some(bounds) = infeasible_bounds {
        return Ok(bounds);
    }

    if let Some(collector) = facet_collector.take() {
        collector.merge_into(input, &mut best_lower, &best_upper, label, config.deadline);
    }

    // Fall back to CROWN only when NaN is present (computation error).
    // Infinite bounds are sound overapproximations from inversion widening
    // (clamp_inverted_best_bounds sets inverted elements to [-inf, +inf]).
    // The previous is_finite() check incorrectly discarded ALL optimization
    // progress when any single element was infinite (#2909, cf. #2854 DAG fix).
    let has_nan = best_lower.iter().any(|v| v.is_nan()) || best_upper.iter().any(|v| v.is_nan());

    if !has_nan {
        // Clamp any inverted intervals from cross-iteration elementwise merge.
        // Reference: alpha-beta-CROWN optimized_bounds.py:943-947 detects but
        // does not correct; we must correct for BoundedTensor validity.
        clamp_inverted_best_bounds(&mut best_lower, &mut best_upper, label);
        BoundedTensor::new_allow_infinite(best_lower, best_upper).map_err(|e| {
            NyError::InternalError(format!(
                "{label} best bounds invalid after inversion widening: {e}"
            ))
        })
    } else {
        warn!("{label}: NaN in best bounds, falling back to plain CROWN (#2909)");
        backend.crown_fallback(input)
    }
}

#[cfg(test)]
#[path = "alpha_crown_loop_regression_tests.rs"]
mod regression_tests;

#[cfg(test)]
#[path = "alpha_crown_loop_start_save_best_tests.rs"]
mod start_save_best_tests;

#[cfg(test)]
#[path = "alpha_crown_loop_hook_tests.rs"]
mod hook_tests;
