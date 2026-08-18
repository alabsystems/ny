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
    clamp_inverted_best_bounds, update_elementwise_best_bounds,
};
use ndarray::{Array1, Array3, ArrayD};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::{any::Any, mem::size_of, time::Instant};
use tracing::{debug, info, warn};

/// Gradient pair: (lower_path_gradients, upper_path_gradients) per ReLU layer.
pub(crate) type DualGradients = (Vec<Array1<f32>>, Vec<Array1<f32>>);

/// Optional gamma probes may no-op on capability/numerical/deadline failures
/// after restoring the authoritative state. Callers treat a deadline as an
/// immediate optimization stop so the last authoritative global bound survives.
/// Invariant/configuration errors outside this allow-list remain authoritative.
pub(crate) fn invprop_gamma_probe_can_noop(error: &NyError) -> bool {
    matches!(
        error,
        NyError::ShapeMismatch { .. }
            | NyError::UnsupportedLayer(_)
            | NyError::UnsupportedOp(_)
            | NyError::UnsupportedConfiguration(_)
            | NyError::NumericalInstability(_)
            | NyError::GpuMemoryExceeded { .. }
            | NyError::CpuMemoryExceeded { .. }
            | NyError::InfeasibleDomain(_)
            | NyError::DeadlineExceeded(_)
    )
}

/// Deterministic, well-mixed Rademacher direction for INVPROP SPSA.
///
/// Using the low bit of affine odd multipliers collapses to a two-direction
/// parity checkerboard. SplitMix64 avalanche and a high output bit give each
/// `(iteration, parameter)` pair an independent-looking reproducible sign.
pub(crate) fn invprop_spsa_sign(iter: usize, parameter: usize) -> f32 {
    let mut mixed = (iter as u64)
        .wrapping_mul(0xD1B5_4A32_D192_ED03)
        .wrapping_add((parameter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(0x94D0_49BB_1331_11EB);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^= mixed >> 31;
    if mixed >> 63 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Convert a one-sided SPSA finite difference into a bounded trust-region step.
pub(crate) fn invprop_bounded_spsa_step(
    base_score: f64,
    probe_score: f64,
    probe_delta: f64,
    learning_rate: f32,
) -> Option<f64> {
    if !base_score.is_finite()
        || !probe_score.is_finite()
        || !probe_delta.is_finite()
        || probe_delta <= 0.0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
    {
        return None;
    }
    // The SPSA slope is Δobjective / Δparameter. Dividing by |base_score|
    // suppresses the widest unresolved regions (exactly where gamma must move
    // farthest), so normalize by the known probe displacement and clamp the
    // resulting slope before applying the configured trust radius.
    let direction = ((probe_score - base_score) / probe_delta).clamp(-1.0, 1.0);
    let step = f64::from(learning_rate) * direction;
    step.is_finite().then_some(step)
}

/// Build one projected SPSA update while retaining output-row objective
/// provenance.
///
/// With per-output gammas, every parameter column uses only that output row's
/// finite gap response. Thus progress on a currently non-winning row is not
/// erased by `max_i(gap_i)`. With shared gammas, the one column affects every
/// output, so it uses the mean of per-row normalized responses; this prevents a
/// large-scale row from dominating solely because of units.
pub(crate) fn invprop_projected_spsa_update(
    base_params: &Array3<f32>,
    base_row_gaps: &[Option<f64>],
    probe_row_gaps: &[Option<f64>],
    probe_delta: f64,
    learning_rate: f32,
    iter: usize,
) -> Option<Array3<f32>> {
    let (bound_dim, num_constraints, neuron_dim) = base_params.dim();
    if bound_dim != 2
        || num_constraints == 0
        || neuron_dim == 0
        || base_row_gaps.is_empty()
        || base_row_gaps.len() != probe_row_gaps.len()
        || base_params
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return None;
    }

    let row_step = |base: Option<f64>, probe: Option<f64>| {
        base.zip(probe).and_then(|(base, probe)| {
            invprop_bounded_spsa_step(base, probe, probe_delta, learning_rate)
        })
    };
    let steps: Vec<Option<f64>> = if neuron_dim == 1 {
        if !probe_delta.is_finite()
            || probe_delta <= 0.0
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
        {
            return None;
        }
        let responses: Vec<f64> = base_row_gaps
            .iter()
            .zip(probe_row_gaps)
            .filter_map(|(&base, &probe)| {
                base.zip(probe).and_then(|(base, probe)| {
                    if !base.is_finite() || !probe.is_finite() {
                        return None;
                    }
                    let response = ((probe - base) / probe_delta).clamp(-1.0, 1.0);
                    response.is_finite().then_some(response)
                })
            })
            .collect();
        if responses.is_empty() {
            return None;
        }
        let mean_response = responses.iter().sum::<f64>() / responses.len() as f64;
        let step = f64::from(learning_rate) * mean_response;
        vec![step.is_finite().then_some(step)]
    } else {
        if neuron_dim != base_row_gaps.len() {
            return None;
        }
        base_row_gaps
            .iter()
            .zip(probe_row_gaps)
            .map(|(&base, &probe)| row_step(base, probe))
            .collect()
    };

    let mut updated = base_params.clone();
    for (idx, value) in updated.iter_mut().enumerate() {
        let output_idx = idx % neuron_dim;
        let Some(step) = steps[output_idx] else {
            continue;
        };
        *value =
            (f64::from(*value) + f64::from(invprop_spsa_sign(iter, idx)) * step).max(0.0) as f32;
    }
    if updated.iter().any(|value| !value.is_finite())
        || updated
            .iter()
            .zip(base_params)
            .all(|(candidate, base)| candidate.to_bits() == base.to_bits())
    {
        return None;
    }
    Some(updated)
}

pub(crate) fn native_invprop_seed_treatment_eligible(
    alpha_state: &AlphaState,
    actual_output_dim: usize,
) -> bool {
    if actual_output_dim == 0 {
        return false;
    }
    let Some(state) = alpha_state.invprop_state.as_ref() else {
        return false;
    };
    let constraints = &state.constraints;
    if !constraints.is_conjunction
        || constraints.clause_indices.is_some()
        || constraints.num_constraints() == 0
        || constraints.output_dim() != actual_output_dim
        || constraints.rhs.len() != constraints.num_constraints()
        || constraints
            .a_matrix
            .iter()
            .chain(constraints.rhs.iter())
            .any(|value| !value.is_finite())
    {
        return false;
    }
    state
        .layer_gammas(crate::invprop::INVPROP_OUTPUT_SEED)
        .is_some_and(|gammas| {
            gammas.active
                && !gammas.gammas.is_empty()
                && gammas.gammas.shape()[0] == 2
                && gammas.num_constraints() == constraints.num_constraints()
                && (gammas.num_neurons() == 1 || gammas.num_neurons() == actual_output_dim)
                && gammas
                    .gammas
                    .iter()
                    .all(|gamma| gamma.is_finite() && *gamma >= 0.0)
        })
}

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
/// explicitly enabled, default-dark gamma optimization (see the throughput
/// guard at the call site).
const INVPROP_ASCENT_MAX_ITERS: usize = 5;
/// Pure gamma-only Linear/ReLU-stable episodes have no alpha-gradient cost and
/// each backward is cheap. Give row-wise SPSA enough mixed directions to solve
/// coupled polyhedral emptiness that five probes routinely miss.
const INVPROP_GAMMA_ONLY_ASCENT_MAX_ITERS: usize = 20;

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
/// feasible conditioned bounds are excluded from the global best-box merge; a
/// gamma iterate affects the verdict only through a typed finite-inversion
/// certificate. The gradient therefore need not be exact or certified; a
/// deterministic bounded one-sided SPSA estimate is sufficient to steer the
/// search without weakening proof authority.
///
/// The extra backward is normally a discarded gradient probe. If its typed
/// pre-repair result certifies a finite inversion, the installed perturbed
/// gamma state and fold attribution are promoted atomically and the caller may
/// terminate with the infeasibility sentinel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InvpropGammaStepOutcome {
    Continue,
    CertifiedInfeasible,
    DeadlineExceeded,
}

fn invprop_seed_gamma_ascent_step<B: AlphaCrownBackend>(
    backend: &B,
    config: &AlphaCrownConfig,
    alpha_state: &mut AlphaState,
    input: &BoundedTensor,
    iter: usize,
    base_row_gaps: &[Option<f64>],
    actual_output_dim: usize,
) -> Result<InvpropGammaStepOutcome> {
    if base_row_gaps.len() != actual_output_dim || !base_row_gaps.iter().any(Option::is_some) {
        return Ok(InvpropGammaStepOutcome::Continue);
    }
    if !native_invprop_seed_treatment_eligible(alpha_state, actual_output_dim) {
        return Ok(InvpropGammaStepOutcome::Continue);
    }
    // Snapshot the current seed duals.
    let seed = match alpha_state
        .invprop_state
        .as_ref()
        .and_then(|s| s.layer_gammas(crate::invprop::INVPROP_OUTPUT_SEED))
    {
        Some(g) if g.active && !g.gammas.is_empty() => g.gammas.clone(),
        _ => return Ok(InvpropGammaStepOutcome::Continue),
    };
    crate::execution_telemetry::record_invprop_gamma_step_attempted();

    let lr = if config.invprop.gamma_lr.is_finite() && config.invprop.gamma_lr > 0.0 {
        config.invprop.gamma_lr
    } else {
        0.5
    };
    let delta = 0.1f32; // SPSA probe magnitude

    let restore = |alpha_state: &mut AlphaState, g: &Array3<f32>| {
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
        *v = (*v + delta * invprop_spsa_sign(iter, idx)).max(0.0);
    }
    restore(alpha_state, &perturbed);

    let evaluated_probe_scope = crate::execution_telemetry::begin_invprop_evaluated_fold_scope();
    let probe_result = backend.backward_iteration(alpha_state, input, iter, true, false);
    let probe_concretized = match probe_result {
        Ok(Some(bp)) => bp.linear_bounds.concretize_sound_with_infeasibility(input),
        Ok(None) => {
            restore(alpha_state, &seed);
            return Ok(InvpropGammaStepOutcome::Continue);
        }
        Err(error) => {
            restore(alpha_state, &seed);
            if matches!(error, NyError::DeadlineExceeded(_)) {
                return Ok(InvpropGammaStepOutcome::DeadlineExceeded);
            }
            if invprop_gamma_probe_can_noop(&error) {
                return Ok(InvpropGammaStepOutcome::Continue);
            }
            return Err(error);
        }
    };

    let perturbed_changed = perturbed
        .iter()
        .zip(seed.iter())
        .any(|(candidate, base)| candidate.to_bits() != base.to_bits());
    if probe_concretized.certified_finite_inversion && perturbed_changed {
        crate::execution_telemetry::record_invprop_gamma_step_applied();
        evaluated_probe_scope.commit();
        return Ok(InvpropGammaStepOutcome::CertifiedInfeasible);
    }
    let probe_row_gaps = probe_concretized.row_finite_gaps;
    drop(evaluated_probe_scope);

    // The probe is a transaction: restore the authoritative base state before
    // inspecting its objective or constructing a proposed update. Every early
    // return below therefore leaves the evaluated gamma vector installed.
    restore(alpha_state, &seed);

    // Ascent from the ORIGINAL duals; projection >= 0. Per-output gammas use
    // their own row response, while a shared column uses the mean normalized
    // response across finite rows.
    let Some(updated) = invprop_projected_spsa_update(
        &seed,
        base_row_gaps,
        &probe_row_gaps,
        f64::from(delta),
        lr,
        iter,
    ) else {
        return Ok(InvpropGammaStepOutcome::Continue);
    };
    restore(alpha_state, &updated);
    crate::execution_telemetry::record_invprop_gamma_step_applied();
    Ok(InvpropGammaStepOutcome::Continue)
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
    let mut best_invprop_gap: Option<f64> = None;
    let mut prev_best_invprop_gap: Option<f64> = None;
    let mut no_improve_iters = 0usize;
    let mut lr = config.learning_rate;
    let mut infeasible_bounds: Option<BoundedTensor> = None;
    let mut total_gradient_skips: usize = 0;
    let mut facet_collector =
        AlphaFacetBankCollector::new(best_lower.len(), input.len(), facet_settings);
    if let Some(collector) = facet_collector.as_mut() {
        collector.capture_constant_lower(crown_bounds.lower());
    }
    let invprop_backend_active = invprop_enabled
        && config.invprop.optimize_gammas
        && native_invprop_seed_treatment_eligible(alpha_state, best_lower.len());

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
        // Nonzero INVPROP duals condition the next backward on the assumed
        // violation region. They may prove that region infeasible, but a
        // feasible conditioned result is not a global box bound and cannot be
        // retained by this public bound-returning API.
        let returnable_box_iterate = alpha_state.invprop_state.as_ref().is_none_or(|state| {
            state
                .all_ny_params()
                .iter()
                .all(|gamma| gamma.to_bits() == 0.0_f32.to_bits())
        });

        // Run backward pass.
        let evaluated_fold_scope = invprop_backend_active
            .then(crate::execution_telemetry::begin_invprop_evaluated_fold_scope);
        // OFF keeps exact-zero gamma and therefore follows the ordinary
        // backward route. Only active optimization needs the output-seed fold
        // and its proof provenance.
        let bp_result = match backend.backward_iteration(
            alpha_state,
            input,
            iter,
            invprop_backend_active,
            need_grad,
        )? {
            Some(result) => result,
            None => return backend.crown_fallback(input),
        };

        // Coefficient retention is optional and can copy a large matrix.  If the
        // backward itself crossed the deadline, preserve its scalar result but do
        // not spend post-deadline time materializing another trajectory facet.
        if returnable_box_iterate && !config.past_deadline() {
            if let Some(collector) = facet_collector.as_mut() {
                collector.capture(&bp_result.linear_bounds);
            }
        }

        // Concretize to get actual bounds
        let actual_output_dim = bp_result.linear_bounds.num_outputs();
        let gamma_treatment_admissible =
            native_invprop_seed_treatment_eligible(alpha_state, actual_output_dim);
        let gamma_optimization_active = invprop_backend_active && gamma_treatment_admissible;
        // Row-wise pre-repair provenance allocates and scans one entry per
        // output. Keep ordinary/non-treatment/OFF iterations on the public
        // allocation-free concretization path; probes and active treatment
        // iterations retain the typed proof metadata.
        let (
            mut concrete_bounds,
            certified_finite_inversion,
            invprop_gap_score,
            invprop_row_gap_scores,
        ) = if gamma_optimization_active {
            let concretized = bp_result
                .linear_bounds
                .concretize_sound_with_infeasibility(input);
            (
                concretized.bounds,
                concretized.certified_finite_inversion,
                concretized.max_finite_gap,
                concretized.row_finite_gaps,
            )
        } else {
            (
                bp_result.linear_bounds.concretize_sound(input),
                false,
                None,
                Vec::new(),
            )
        };
        if let Some(bounds_no_oc) = bp_result.bounds_without_oc {
            let no_oc_bounds = bounds_no_oc.concretize_sound(input);
            concrete_bounds = take_best_bounds(&concrete_bounds, &no_oc_bounds);
        }
        if let Some(scope) = evaluated_fold_scope {
            scope.commit();
        }

        // INVPROP infeasibility check
        if let Some(ref mut state) = alpha_state.invprop_state {
            if certified_finite_inversion {
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
        if returnable_box_iterate && config.should_save_best(iter, is_last_iter) {
            update_elementwise_best_bounds(
                &mut best_lower,
                &mut best_upper,
                &concrete_bounds,
                iter,
            )?;
        }

        // Finite-only sum for early stopping (#2857). Layout-agnostic (#1939).
        let lower_sum: f32 = finite_lower_sum(concrete_bounds.lower());
        let gamma_only_invprop = gamma_treatment_admissible && alpha_state.num_unstable() == 0;
        let gamma_probe_limit = if gamma_only_invprop {
            INVPROP_GAMMA_ONLY_ASCENT_MAX_ITERS
        } else {
            INVPROP_ASCENT_MAX_ITERS
        };
        let gamma_gap_control = gamma_optimization_active && invprop_gap_score.is_some();
        if let Some(gap) = invprop_gap_score {
            if best_invprop_gap.is_none_or(|best| gap > best) {
                best_invprop_gap = Some(gap);
            }
        }

        // NaN detection: if any bound element is NaN, the backward pass produced
        // garbage. Break early to avoid wasting remaining iterations — the
        // post-loop has_nan check will fall back to CROWN. (#2597)
        if concrete_bounds.lower().iter().any(|v| v.is_nan())
            || concrete_bounds.upper().iter().any(|v| v.is_nan())
        {
            warn!("{label}: NaN in bounds at iteration {iter}, aborting optimization (#2597)");
            break;
        }

        // A full authoritative backward can itself consume the remaining
        // budget. Preserve any typed proof handled above and any returnable
        // global iterate, then stop before refresh/gradient probe work.
        if config.past_deadline() {
            if returnable_box_iterate && !config.should_save_best(iter, false) {
                update_elementwise_best_bounds(
                    &mut best_lower,
                    &mut best_upper,
                    &concrete_bounds,
                    iter,
                )?;
            }
            info!("{label}: deadline exceeded after authoritative iteration {iter}");
            break;
        }

        // Track best lower_sum for early stopping
        let improved_output = lower_sum > best_lower_sum;
        if improved_output {
            best_lower_sum = lower_sum;
        }

        // Early stopping check (compare best improvement since last iteration).
        let best_improvement = if gamma_gap_control {
            match (best_invprop_gap, prev_best_invprop_gap) {
                (Some(best), Some(previous)) => best - previous,
                (Some(_), None) => f64::INFINITY,
                _ => 0.0,
            }
        } else {
            f64::from(best_lower_sum - prev_best_lower_sum)
        };
        if best_improvement < f64::from(config.tolerance) {
            no_improve_iters += 1;
        } else {
            no_improve_iters = 0;
        }
        let gamma_probe_available = gamma_optimization_active
            && invprop_gap_score.is_some()
            && need_grad
            && iter < gamma_probe_limit
            && !config.past_deadline();
        if iter > 0 && no_improve_iters >= config.early_stop_patience && !gamma_probe_available {
            // Force-save before early exit to avoid losing optimization progress
            // when patience is exhausted during the warmup window.
            // Reference: optimized_bounds.py:794 (patience == early_stop_patience).
            if returnable_box_iterate && !config.should_save_best(iter, false) {
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
        if iter == 1
            && !gamma_gap_control
            && backend.pilot_check(config, best_lower_sum, &crown_bounds)
        {
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

        // INVPROP probes the exact alpha state that produced this iteration's
        // typed gap. Run it before reference refresh or alpha-gradient work so
        // gamma-only progress and direct proof promotion do not pay for
        // unrelated discarded backwards.
        let mut gamma_step_outcome = InvpropGammaStepOutcome::Continue;
        if gamma_probe_available {
            if invprop_gap_score.is_some() {
                gamma_step_outcome = invprop_seed_gamma_ascent_step(
                    &*backend,
                    config,
                    alpha_state,
                    input,
                    iter,
                    &invprop_row_gap_scores,
                    actual_output_dim,
                )?;
                if gamma_step_outcome == InvpropGammaStepOutcome::CertifiedInfeasible {
                    if let Some(state) = alpha_state.invprop_state.as_mut() {
                        state.mark_infeasible(0)?;
                        state.apply_infeasible_mask(&mut concrete_bounds);
                        infeasible_bounds = Some(concrete_bounds);
                        break;
                    }
                }
            }
        }

        if gamma_step_outcome == InvpropGammaStepOutcome::DeadlineExceeded {
            info!("{label}: INVPROP probe reached the deadline at iteration {iter}");
            break;
        }

        if config.past_deadline() {
            info!("{label}: deadline exceeded by INVPROP probe at iteration {iter}");
            break;
        }

        // A pure-linear native route has no alpha or extended parameters to
        // update. OFF therefore needs exactly its authoritative identity fold;
        // ON continues only while another bounded gamma probe can change the
        // next evaluated seed, then stops after one final authoritative fold.
        if gamma_only_invprop {
            if !config.invprop.optimize_gammas
                || invprop_gap_score.is_none()
                || iter >= gamma_probe_limit
            {
                break;
            }
            prev_best_lower_sum = best_lower_sum;
            prev_best_invprop_gap = best_invprop_gap;
            continue;
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
        prev_best_invprop_gap = best_invprop_gap;
    }
    // Gradient skip summary (#2981 Slice 5).
    if total_gradient_skips > 0 {
        warn!(
            "{label}: skipped {total_gradient_skips}/{} gradient updates (non-finite)",
            config.iterations * alpha_state.alphas.len()
        );
    }

    // Gamma-conditioned iterates are deliberately excluded from the returned
    // global box. If alpha moved while gamma was nonzero but did not obtain a
    // proof, recover that valid alpha progress with one deadline-gated fold at
    // exact zero output-seed gamma. The seed transaction is restored on every
    // outcome; optional backend/deadline failures simply retain the initial
    // global CROWN bound already stored in `best_*`.
    if infeasible_bounds.is_none()
        && invprop_enabled
        && config.invprop.optimize_gammas
        && alpha_state.num_unstable() > 0
        && !config.past_deadline()
        && native_invprop_seed_treatment_eligible(alpha_state, best_lower.len())
    {
        let seed = alpha_state
            .invprop_state
            .as_ref()
            .and_then(|state| state.layer_gammas(crate::invprop::INVPROP_OUTPUT_SEED))
            .map(|gammas| gammas.gammas.clone());
        if let Some(seed) = seed.filter(|values| {
            values
                .iter()
                .any(|value| value.to_bits() != 0.0_f32.to_bits())
        }) {
            if let Some(gammas) = alpha_state
                .invprop_state
                .as_mut()
                .and_then(|state| state.layer_gammas_mut(crate::invprop::INVPROP_OUTPUT_SEED))
            {
                gammas.gammas.fill(0.0);
            }
            let globally_unconditioned = alpha_state.invprop_state.as_ref().is_some_and(|state| {
                state
                    .all_ny_params()
                    .iter()
                    .all(|value| value.to_bits() == 0.0_f32.to_bits())
            });
            let final_result = globally_unconditioned.then(|| {
                backend.backward_iteration(
                    alpha_state,
                    input,
                    config.iterations,
                    // Exact-zero gamma is the identity; use the ordinary
                    // backend route for this global reconstruction.
                    false,
                    false,
                )
            });
            if let Some(gammas) = alpha_state
                .invprop_state
                .as_mut()
                .and_then(|state| state.layer_gammas_mut(crate::invprop::INVPROP_OUTPUT_SEED))
            {
                gammas.gammas.assign(&seed);
            }
            match final_result {
                Some(Ok(Some(bp))) => {
                    let mut global_bounds = bp.linear_bounds.concretize_sound(input);
                    if let Some(bounds_no_oc) = bp.bounds_without_oc {
                        global_bounds =
                            take_best_bounds(&global_bounds, &bounds_no_oc.concretize_sound(input));
                    }
                    if !global_bounds.lower().iter().any(|value| value.is_nan())
                        && !global_bounds.upper().iter().any(|value| value.is_nan())
                    {
                        update_elementwise_best_bounds(
                            &mut best_lower,
                            &mut best_upper,
                            &global_bounds,
                            config.iterations,
                        )?;
                    }
                }
                Some(Ok(None)) | None => {}
                Some(Err(error)) if invprop_gamma_probe_can_noop(&error) => {}
                Some(Err(error)) => {
                    return Err(error);
                }
            }
        }
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
