// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DAG α-CROWN propagation for `GraphNetwork`.
//!
//! Contains the α-CROWN implementation for non-sequential (DAG) graphs like ResNet
//! with skip connections and multiple paths.

mod alpha_update;
mod collect;
mod diagnostics;
// #[cfg(test)] mod fl_composition_tests;
//
// DECLARATION WITHDRAWN 2026-08-02: `a0e388bd` added this `mod` but never committed
// `fl_composition_tests.rs` — the file exists on no branch
// (`git log --all --diff-filter=A` finds nothing). Because it is `#[cfg(test)]` the
// LIB still builds, so the gap is invisible to `cargo build`; it breaks only
// `cargo test -p ny-propagate`, which fails with
//
//     error[E0583]: file not found for module `fl_composition_tests`
//
// i.e. this crate's ~8,900-test suite — including its soundness oracles — could not
// compile at all and therefore could not gate anything.
//
// Re-land the missing file and this line together, in one commit.
mod gradients;
mod init;
#[cfg(test)]
mod iter0_parity_tests;
mod preloop_cuda_rows;
mod spec_axis;
mod supplements;

use init::{DagAlphaInitResult, DagAlphaInitState};
use preloop_cuda_rows::PreloopCudaRowsOutcome;

use crate::bounds::{AlphaCrownConfig, AlphaSpecAscent};

use crate::bounds::GraphAlphaState;
use ndarray::{Array1, Array3, ArrayD};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info, instrument, warn};

use super::propagate_helpers::{clamp_inverted_best_bounds, update_elementwise_best_bounds};
use super::resnet_decompose::{multiobj_joint_alpha_enabled, root_alpha_gpu_enabled};
use super::resnet_skeleton::{build_resnet_segment_skeleton, extract_skeleton_enabled};
use super::runtime_state::DagAlphaRuntimeState;
use crate::network::alpha_crown_loop::{
    alpha_iteration_needs_gradient, final_alpha_bound_only_enabled, finite_lower_sum,
    invprop_gamma_probe_can_noop, invprop_projected_spsa_update, invprop_spsa_sign,
};
use crate::network::core::GraphNetwork;
use crate::network::graph_alpha::bounds::AlphaReferenceBoundsSource;

#[cfg(test)]
thread_local! {
    /// Successful zero-gamma recovery folds that strictly tighten at least one
    /// endpoint of the pre-recovery global best. Test-only route evidence.
    pub(super) static INVPROP_ZERO_GAMMA_RECOVERY_IMPROVEMENTS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

fn install_invprop_seed_params(
    runtime: &mut DagAlphaRuntimeState,
    seed_key: &str,
    params: &Array3<f32>,
) -> Result<()> {
    let gammas = runtime
        .invprop_mut()
        .ok_or_else(|| {
            NyError::InternalError(
                "DAG INVPROP gamma transaction lost its initialized state".to_string(),
            )
        })?
        .layer_gammas_mut(seed_key)
        .ok_or_else(|| {
            NyError::InternalError("DAG INVPROP gamma transaction lost output seed".to_string())
        })?;
    if gammas.gammas.dim() != params.dim() {
        return Err(NyError::InternalError(format!(
            "DAG INVPROP seed shape changed during transaction: {:?} != {:?}",
            gammas.gammas.dim(),
            params.dim()
        )));
    }
    gammas.gammas.assign(params);
    Ok(())
}

fn params_bit_identical(lhs: &Array3<f32>, rhs: &Array3<f32>) -> bool {
    lhs.dim() == rhs.dim()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn invprop_output_seed_treatment_eligible(config: &AlphaCrownConfig, output_dim: usize) -> bool {
    config.iterations > 0
        && config.invprop.enabled
        && output_dim > 0
        && config
            .output_constraints
            .as_ref()
            .is_some_and(|constraints| {
                constraints.is_conjunction
                    && constraints.clause_indices.is_none()
                    && constraints.num_constraints() > 0
                    && constraints.output_dim() == output_dim
                    && constraints.rhs.len() == constraints.num_constraints()
                    && constraints
                        .a_matrix
                        .iter()
                        .chain(constraints.rhs.iter())
                        .all(|value| value.is_finite())
            })
        && crate::output_margin_seed::margin_subset_indices(output_dim).is_none()
}

fn uses_patches_output_seed(network: &GraphNetwork, output_bounds: &BoundedTensor) -> bool {
    output_bounds.shape().len() == 3
        && network.use_patches_mode
        && network
            .nodes
            .values()
            .any(|node| matches!(node.layer, crate::layers::Layer::Conv2d(_)))
}

/// Whether the caller consumes the post-loop alpha/optimizer state.
///
/// The final gradient/update is dead only for [`Self::BoundsOnly`]. Collection
/// returns the state as a BaB warm start and immediately re-evaluates it, so
/// skipping that update would change an observable result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DagAlphaLoopResultUse {
    BoundsOnly,
    BoundsAndState,
}

impl DagAlphaLoopResultUse {
    fn terminal_bound_only(self, gate_enabled: bool) -> bool {
        gate_enabled && matches!(self, Self::BoundsOnly)
    }
}

/// The root checkpoint policy may recover only an owned artifact containing at
/// least one finite, completely returned fold.  Other errors, and any deadline
/// before that boundary, retain the legacy fail-closed behavior.
fn retain_completed_deadline_error(
    policy_enabled: bool,
    completed_iterations: usize,
    error: &NyError,
) -> bool {
    policy_enabled && completed_iterations > 0 && matches!(error, NyError::DeadlineExceeded(_))
}

/// Internal result of one DAG alpha optimization episode.
///
/// `reference_bounds` is the optimizer's final
/// [`GraphAlphaReferenceBounds::current`](super::reference_bounds::GraphAlphaReferenceBounds::current)
/// map. Collection callers can therefore re-evaluate the returned alpha state
/// without silently starting a second intermediate-bound transaction.
pub(in crate::network::graph_alpha) struct DagAlphaCollectionArtifact {
    pub(in crate::network::graph_alpha) output_bounds: BoundedTensor,
    pub(in crate::network::graph_alpha) alpha_state: GraphAlphaState,
    pub(in crate::network::graph_alpha) reference_bounds:
        std::collections::HashMap<String, BoundedTensor>,
    /// Actual reference collector used by initialization. This is carried
    /// through the optimizer so post-alpha dispatch never infers engagement
    /// from a request that may have failed closed to an ordinary collector.
    pub(in crate::network::graph_alpha) reference_bounds_source: AlphaReferenceBoundsSource,
    /// Number of alpha iterates whose complete bound fold returned and passed
    /// the NaN publication check.  A phase-cap checkpoint may be minted only
    /// after this is nonzero; an expiry before the first completed fold keeps
    /// the historical `DeadlineExceeded` contract.
    pub(in crate::network::graph_alpha) completed_iterations: usize,
    /// Number of alpha optimizer updates represented by `alpha_state`.  This
    /// can differ from completed bound folds when a deadline lands between a
    /// fold and its update, or when best-margin selection restores an earlier
    /// whole-state snapshot.
    pub(in crate::network::graph_alpha) optimizer_updates_completed: usize,
}

/// #alpha-zero-yield gate (invariant I3). `NY_ALPHA_ZERO_YIELD_FRAC=<0..0.9>`;
/// where the env var is absent the preset default
/// (`AlphaCrownConfig::alpha_zero_yield_frac`, typed key
/// `alpha_crown.alpha_zero_yield_frac`) applies; both absent = off =
/// byte-identical. A FRACTION of the ascent's own window, never a fixed number
/// of seconds or iterations -- that is invariant I1.
///
/// MEASURED (2026-08-11, official 100 s budget,
/// docs/LEVER_CENSUS_AND_ROOT_ALPHA_REMEASURE_2026-08-11.md §8 + addendum):
/// on a 16-row cifar100_medium sample, fires on 15/15 timeout rows, returns
/// 8.4-14.8 s of root time per row (mean ~10.1 s), and moves root-verified
/// objectives +15/+1/+1 on three rows with 0 regressions and 0 verdict
/// changes. A subsequent 16-row cifar100_large sample did not engage the gate.
/// The original runs retained neither a complete per-row artifact nor a
/// row-complete A/B for the 200-row preset, so shipped presets remain unset.
fn parse_alpha_zero_yield_frac(raw: &std::ffi::OsStr) -> Option<f64> {
    raw.to_str()?
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && (0.0..0.9).contains(v) && *v > 0.0)
}

/// Raw env presence, latched once per process (the `OsString`, not the decision,
/// so per-run preset defaults still vary run-to-run).
fn alpha_zero_yield_env_raw() -> Option<&'static std::ffi::OsStr> {
    static F: std::sync::OnceLock<Option<std::ffi::OsString>> = std::sync::OnceLock::new();
    F.get_or_init(|| std::env::var_os("NY_ALPHA_ZERO_YIELD_FRAC"))
        .as_deref()
}

/// Env wins wherever PRESENT, in both directions: a valid value arms it over a
/// silent preset, and an invalid value (`0`, `off`, garbage) is a kill switch
/// for a preset-armed fraction — the #root-alpha-margin idiom.
fn alpha_zero_yield_frac_from(
    raw: Option<&std::ffi::OsStr>,
    preset_default: Option<f64>,
) -> Option<f64> {
    match raw {
        Some(raw) => parse_alpha_zero_yield_frac(raw),
        None => preset_default.filter(|v| AlphaCrownConfig::alpha_zero_yield_frac_is_valid(*v)),
    }
}

fn alpha_zero_yield_frac(config: &AlphaCrownConfig) -> Option<f64> {
    alpha_zero_yield_frac_from(alpha_zero_yield_env_raw(), config.alpha_zero_yield_frac)
}

/// Parse the #root-alpha-margin environment override. Only exact `"1"` arms;
/// every other present value is a kill switch for the preset-supplied default.
fn parse_root_alpha_margin(raw: Option<&str>) -> bool {
    raw == Some("1")
}

fn root_alpha_margin_enabled_from(raw: Option<&std::ffi::OsStr>, preset_default: bool) -> bool {
    match raw {
        None => preset_default,
        Some(raw) => parse_root_alpha_margin(raw.to_str()),
    }
}

/// Resolve #root-alpha-margin with a PRESET-supplied default.
///
/// Delivery contract (see `crates/ny-cli/tests/measured_gate_delivery.rs`): an experimental,
/// default-off lever needs a typed preset key to be available to a future scored run, because
/// the scored wrapper exports exactly one `NY_*` variable. `AlphaCrownPreset::root_alpha_margin`
/// is that key.
///
/// The env var still WINS wherever it is present, in both directions: `=1` arms a lever the
/// preset did not ask for, and `=0` (or anything malformed) kills one the preset did. That
/// keeps a single kill switch for a scored run regardless of which yaml is loaded.
pub(crate) fn root_alpha_margin_enabled_with(preset_default: bool) -> bool {
    let raw = std::env::var_os("NY_ROOT_ALPHA_MARGIN");
    root_alpha_margin_enabled_from(raw.as_deref(), preset_default)
}

fn root_alpha_margin_state_from<'a>(
    config: &'a AlphaCrownConfig,
    raw: Option<&std::ffi::OsStr>,
) -> (bool, Option<&'a AlphaSpecAscent>) {
    let enabled = root_alpha_margin_enabled_from(raw, config.root_alpha_margin);
    let spec_ascent = enabled.then_some(config.spec_ascent.as_ref()).flatten();
    (enabled, spec_ascent)
}

fn root_alpha_margin_state(config: &AlphaCrownConfig) -> (bool, Option<&AlphaSpecAscent>) {
    let raw = std::env::var_os("NY_ROOT_ALPHA_MARGIN");
    root_alpha_margin_state_from(config, raw.as_deref())
}

/// Subordinate dark gate for the margin-directed warmup gradient.
///
/// The effective parent gate (typed default or `NY_ROOT_ALPHA_MARGIN=1`) owns
/// the spec rows and whole-state checkpoint. Do not even read this child gate
/// unless the parent is armed: setting the child by itself must preserve the
/// legacy warmup byte path.
fn parse_root_alpha_margin_gradient(raw: Option<&str>) -> bool {
    raw == Some("1")
}

fn root_alpha_margin_gradient_enabled_if<F>(parent_enabled: bool, read: F) -> bool
where
    F: FnOnce() -> Option<String>,
{
    parent_enabled && parse_root_alpha_margin_gradient(read().as_deref())
}

fn root_alpha_margin_gradient_enabled(parent_enabled: bool) -> bool {
    root_alpha_margin_gradient_enabled_if(parent_enabled, || {
        std::env::var("NY_ROOT_ALPHA_MARGIN_GRADIENT").ok()
    })
}

fn retain_warmup_iter_cache(root_cache_enabled: bool, margin_gradient_eligible: bool) -> bool {
    root_cache_enabled || margin_gradient_eligible
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MarginGradientEligibility {
    Eligible,
    NotRequested,
    MissingSpecObjective,
    NonAnalyticChain,
    NonReluOptimizerState,
    ConflictMultiobjJointAlpha,
    ConflictRootAlphaTrue,
    ConflictBothAlphaPolicies,
}

impl MarginGradientEligibility {
    fn is_eligible(self) -> bool {
        matches!(self, Self::Eligible)
    }

    fn reason(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::NotRequested => "not_requested",
            Self::MissingSpecObjective => "missing_spec_objective",
            Self::NonAnalyticChain => "non_analytic_chain",
            Self::NonReluOptimizerState => "non_relu_optimizer_state",
            Self::ConflictMultiobjJointAlpha => "conflict_NY_MULTIOBJ_JOINT_ALPHA",
            Self::ConflictRootAlphaTrue => "conflict_NY_ROOT_ALPHA_TRUE",
            Self::ConflictBothAlphaPolicies => {
                "conflict_NY_MULTIOBJ_JOINT_ALPHA_and_NY_ROOT_ALPHA_TRUE"
            }
        }
    }
}

fn margin_gradient_eligibility(
    requested: bool,
    has_spec_objective: bool,
    analytic_chain: bool,
    relu_only_optimizer_state: bool,
    multiobj_joint_alpha: bool,
    root_alpha_true: bool,
) -> MarginGradientEligibility {
    if !requested {
        return MarginGradientEligibility::NotRequested;
    }
    if !has_spec_objective {
        return MarginGradientEligibility::MissingSpecObjective;
    }
    if !analytic_chain {
        return MarginGradientEligibility::NonAnalyticChain;
    }
    if !relu_only_optimizer_state {
        return MarginGradientEligibility::NonReluOptimizerState;
    }
    match (multiobj_joint_alpha, root_alpha_true) {
        (true, true) => MarginGradientEligibility::ConflictBothAlphaPolicies,
        (true, false) => MarginGradientEligibility::ConflictMultiobjJointAlpha,
        (false, true) => MarginGradientEligibility::ConflictRootAlphaTrue,
        (false, false) => MarginGradientEligibility::Eligible,
    }
}

#[derive(Clone, Debug, PartialEq)]
struct BindingMarginObjective {
    row_index: usize,
    slack: f32,
    /// A lower-bound objective. Upper-verification rows are sign-flipped so
    /// ascending `lower(-c^T y)` is exactly descending `upper(c^T y)`.
    lower_objective: Vec<f32>,
}

/// Select the current worst unverified margin row as a direct lower objective.
///
/// This is the first bounded active-objective policy: one row, deterministic
/// first-row tie breaking, and all-or-nothing validation.  A malformed row
/// refuses the complete proposal instead of optimizing a partial conjunction.
/// How far below the running best an α iterate may fall before the ascent is
/// treated as diverged rather than slow. Expressed as a multiple of
/// `max(1, |best|)`, so it scales with the objective and stays meaningful when
/// the best is near zero. `1e6` is far outside anything a converging projected
/// ascent produces and far inside the measured 1e20 blow-up.
const ALPHA_DIVERGENCE_FACTOR: f64 = 1e6;

/// Dark gate for the α divergence bail-out (`NY_ALPHA_DIVERGENCE_BAIL=1`).
/// Only the exact value `"1"` arms it; everything else keeps the shipped
/// patience-only behavior.
fn parse_alpha_divergence_bail(raw: Option<&str>) -> bool {
    raw == Some("1")
}

fn alpha_divergence_bail_enabled() -> bool {
    parse_alpha_divergence_bail(std::env::var("NY_ALPHA_DIVERGENCE_BAIL").ok().as_deref())
}

/// Dark sub-gate for hinge-subgradient steering (`NY_ROOT_ALPHA_MARGIN_HINGE=1`).
/// Subordinate to the margin-gradient child gate, which is itself subordinate to
/// `NY_ROOT_ALPHA_MARGIN`. Only the exact value `"1"` arms it.
fn parse_root_alpha_margin_hinge(raw: Option<&str>) -> bool {
    raw == Some("1")
}

fn root_alpha_margin_hinge_enabled(gradient_enabled: bool) -> bool {
    gradient_enabled
        && parse_root_alpha_margin_hinge(
            std::env::var("NY_ROOT_ALPHA_MARGIN_HINGE").ok().as_deref(),
        )
}

/// Steer by the HINGE SUBGRADIENT — the sum of every currently-violated row's
/// objective — instead of by the single worst row.
///
/// WHY. Selection and steering currently optimize *different* objectives:
/// [`AlphaSpecAscent::hinge_score`] ranks iterates by `Σ_r min(0, slack_r)` over
/// all unproven rows, while [`binding_margin_lower_objective`] points the
/// gradient at the single most-violated row. Greedy single-row steering
/// oscillates on a 99-row conjunction — tighten the worst row, a different row
/// becomes worst, and the first regresses — which is what the measured plateau
/// after ~5 productive iterations looks like.
///
/// `Σ_r min(0, slack_r)` is concave-piecewise-linear in the output bound, and
/// its subgradient is exactly the **unweighted sum of the objectives of the
/// currently violated rows** (each violated row contributes `∂slack_r`, each
/// satisfied row contributes 0). Summing them makes the direction the ascent
/// moves agree with the score it is judged by.
///
/// `row_index`/`slack` are retained for telemetry and report the worst
/// contributing row, so existing log lines stay meaningful.
///
/// SOUND: steering only. Any `α ∈ [0,1]` yields a valid bound, selection still
/// keeps the best-scoring iterate, so no direction can produce a wrong verdict.
fn hinge_margin_lower_objective(
    ascent: &AlphaSpecAscent,
    lower: &[f32],
    upper: &[f32],
) -> Option<BindingMarginObjective> {
    let width = ascent.output_len();
    if width == 0 {
        return None;
    }
    let mut acc = vec![0.0f32; width];
    let mut worst: Option<(usize, f32)> = None;
    let mut contributors = 0usize;
    for (row_index, row) in ascent.rows.iter().enumerate() {
        let slack = row.margin_slack(lower, upper)?;
        if slack > 0.0 {
            continue;
        }
        if row.objective.len() != width {
            return None;
        }
        for (slot, value) in acc.iter_mut().zip(&row.objective) {
            // Upper-verification rows are sign-flipped so that ascending
            // `lower(-cᵀy)` is exactly descending `upper(cᵀy)` — same convention
            // as the single-row path.
            *slot += if row.verify_upper_bound {
                -*value
            } else {
                *value
            };
        }
        contributors += 1;
        if worst.is_none_or(|(_, w)| slack < w) {
            worst = Some((row_index, slack));
        }
    }
    let (row_index, slack) = worst?;
    if contributors == 0 || acc.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(BindingMarginObjective {
        row_index,
        slack,
        lower_objective: acc,
    })
}

fn binding_margin_lower_objective(
    ascent: &AlphaSpecAscent,
    lower: &[f32],
    upper: &[f32],
) -> Option<BindingMarginObjective> {
    let mut binding: Option<BindingMarginObjective> = None;
    for (row_index, row) in ascent.rows.iter().enumerate() {
        let slack = row.margin_slack(lower, upper)?;
        if slack > 0.0 {
            continue;
        }
        if binding
            .as_ref()
            .is_some_and(|current| slack >= current.slack)
        {
            continue;
        }
        let lower_objective: Vec<f32> = if row.verify_upper_bound {
            row.objective.iter().map(|value| -*value).collect()
        } else {
            row.objective.clone()
        };
        if lower_objective.is_empty() || lower_objective.iter().any(|value| !value.is_finite()) {
            return None;
        }
        binding = Some(BindingMarginObjective {
            row_index,
            slack,
            lower_objective,
        });
    }
    binding
}

/// #binding-row-replay trust region (consult #6 v1 acceptance policy, cheap
/// half): the per-iteration cap on |Δα| for REPLAY-sourced margin-gradient
/// updates. Applied as a post-`update_all_alphas` projection; the updater
/// itself is untouched and every other gradient source is byte-identical.
const REPLAY_ALPHA_TRUST_REGION: f32 = 0.05;

/// Pre-update snapshot of every ReLU (lower, upper) α pair, taken only on
/// replay-sourced iterations (#binding-row-replay trust region).
fn snapshot_relu_alpha_pairs(state: &GraphAlphaState) -> Vec<(String, Array1<f32>, Array1<f32>)> {
    let names: Vec<String> = state.relu_nodes().map(str::to_string).collect();
    names
        .into_iter()
        .filter_map(|name| {
            let (lower, upper) = state.relu_alpha_pair(&name)?;
            Some((name, lower.clone(), upper.clone()))
        })
        .collect()
}

/// Project the post-optimizer ReLU α state onto the replay trust region:
/// each component moves at most [`REPLAY_ALPHA_TRUST_REGION`] from its
/// pre-update value (`α ← α_prev + clamp(α − α_prev, ±0.05)`). Both endpoints
/// lie in [0,1] (the updater already projected), so the result needs no
/// second [0,1] projection. Saturated components land EXACTLY at
/// `α_prev ± 0.05`. Width mismatches (an optimizer that resized a vector)
/// skip the node — fail open to the updater's own projection rather than
/// corrupt state. Returns the number of clamped components (telemetry).
fn clamp_relu_alpha_trust_region(
    state: &mut GraphAlphaState,
    snapshot: &[(String, Array1<f32>, Array1<f32>)],
) -> usize {
    let mut clamped = 0usize;
    for (name, prev_lower, prev_upper) in snapshot {
        let Some((lower, upper)) = state.relu_alpha_pair_mut(name) else {
            continue;
        };
        for (current, previous) in [(lower, prev_lower), (upper, prev_upper)] {
            if current.len() != previous.len() {
                continue;
            }
            for (value, &anchor) in current.iter_mut().zip(previous.iter()) {
                let delta = *value - anchor;
                if delta > REPLAY_ALPHA_TRUST_REGION {
                    *value = anchor + REPLAY_ALPHA_TRUST_REGION;
                    clamped += 1;
                } else if delta < -REPLAY_ALPHA_TRUST_REGION {
                    *value = anchor - REPLAY_ALPHA_TRUST_REGION;
                    clamped += 1;
                }
            }
        }
    }
    clamped
}

/// Parse the share of the remaining root budget available to the complete
/// sequence of intermediate alpha-bound refreshes. Invalid or out-of-range
/// values preserve the typed fallback so a malformed measurement override
/// cannot silently disable or monopolize the refresh lane.
fn parse_alpha_refresh_fraction(raw: Option<&str>, fallback: f32) -> f32 {
    raw.and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|&fraction| AlphaCrownConfig::reference_refresh_fraction_is_valid(fraction))
        .unwrap_or(fallback)
}

fn alpha_refresh_fraction(config: &AlphaCrownConfig) -> f32 {
    let configured = config.resolved_reference_refresh_fraction();
    let raw = std::env::var("NY_ALPHA_REFRESH_FRACTION").ok();
    parse_alpha_refresh_fraction(raw.as_deref(), configured)
}

/// Lazily create one cumulative refresh pool and return the allowance for the
/// next refresh.
///
/// `None` means an intentionally unbounded refresh (no global deadline and no
/// configured absolute cap). Otherwise the global remainder can shrink
/// independently, so every allowance is clamped to the aggregate pool and the
/// live verifier deadline. The fraction and absolute cap are applied exactly
/// once: subsequent improving iterations only receive what remains.
fn cumulative_alpha_refresh_allowance(
    budget_remaining: &mut Option<std::time::Duration>,
    global_remaining: Option<std::time::Duration>,
    fraction: f32,
    max_duration: Option<std::time::Duration>,
) -> Option<std::time::Duration> {
    if budget_remaining.is_none() {
        let fractional = global_remaining.map(|remaining| remaining.mul_f32(fraction));
        *budget_remaining = match (fractional, max_duration) {
            (Some(fractional), Some(maximum)) => Some(fractional.min(maximum)),
            (Some(fractional), None) => Some(fractional),
            (None, Some(maximum)) => Some(maximum),
            (None, None) => None,
        };
    }

    (*budget_remaining).map(|budget| {
        global_remaining
            .map(|global| budget.min(global))
            .unwrap_or(budget)
    })
}

/// Charge actual refresh airtime to the cumulative pool.
///
/// A collector can finish just after its local deadline. Saturating subtraction
/// makes that overrun exhaust the pool instead of accidentally creating more
/// refresh time.
fn debit_alpha_refresh_budget(
    budget_remaining: &mut Option<std::time::Duration>,
    elapsed: std::time::Duration,
) {
    if let Some(budget) = budget_remaining.as_mut() {
        *budget = budget.saturating_sub(elapsed);
    }
}

/// Reuse init-collected bounds only when they match (or tighten) the collector
/// graph-CROWN Step 1 would independently select.
fn can_reuse_initial_node_bounds(
    source: AlphaReferenceBoundsSource,
    step1_would_use_forward_linear: bool,
) -> bool {
    match source {
        // This route starts from the complete forward-linear Step-1 map and
        // intersects every demanded CROWN target, so it provably dominates
        // that map element-wise.
        AlphaReferenceBoundsSource::CganCompleteCrownIbp { .. } => step1_would_use_forward_linear,
        // The typed transaction starts from the complete forward-linear map,
        // intersects at most one atomic CROWN target, and applies only
        // shrink-only downstream resweeps. It is therefore a strictly
        // compatible Step-1 authority exactly where Graph-CROWN would have
        // selected the forward-linear map itself. Reusing it also lets the
        // initial output CROWN consume the target tightening instead of
        // recollecting/cloning the weaker cached baseline.
        AlphaReferenceBoundsSource::CganSparseTargetComplete { .. } => {
            step1_would_use_forward_linear
        }
        AlphaReferenceBoundsSource::CrownIbp => !step1_would_use_forward_linear,
        AlphaReferenceBoundsSource::ForwardLinear => step1_would_use_forward_linear,
        AlphaReferenceBoundsSource::Ibp => false,
    }
}

/// Immutable context for the DAG alpha optimization loop.
///
/// Bundles parameters that are fixed for the entire optimization. Mutable state
/// (`runtime`, `bilinear_alphas`, etc.) is passed as separate `&mut` parameters
/// so the borrow checker can reason about independent borrows.
pub(super) struct DagAlphaLoopContext<'a> {
    pub(super) input: &'a BoundedTensor,
    pub(super) exec_order: &'a [String],
    pub(super) output_dim: usize,
    pub(super) input_dim: usize,
    pub(super) config: &'a AlphaCrownConfig,
    pub(super) engine: Option<&'a dyn GemmEngine>,
    /// #alpha-steering-proposal: the dedicated α-gradient PROPOSAL channel
    /// (`crate::alpha_gradient_steering`), populated only when the
    /// margin-gradient lane is armed. DISTINCT from `engine` by contract: it
    /// must never be consulted by a bound/precheck/BaB call site, and its one
    /// consumer (`try_wgpu_proposal_joint_gradients`) dispatches an API that
    /// returns gradients only. Never assign it to `engine`.
    pub(super) alpha_steering: Option<&'a dyn GemmEngine>,
    pub(super) relu_nodes: &'a [(String, usize)],
    pub(super) has_bilinear: bool,
    pub(super) has_mul_binary: bool,
}

impl GraphNetwork {
    /// α-CROWN for DAG (non-sequential) graphs like ResNet with optional GEMM acceleration.
    ///
    /// This handles graphs with skip connections (Add operations) and multiple paths.
    /// The backward pass accumulates linear bounds from all consumers of each node.
    #[instrument(skip(self, input, config, engine), fields(num_nodes = self.nodes.len(), iterations = config.iterations))]
    pub(super) fn propagate_dag_alpha_crown_with_config_and_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let init = match self.init_dag_alpha_state(input, config, engine, None)? {
            DagAlphaInitResult::EarlyReturn { bounds, .. } => return Ok(bounds),
            DagAlphaInitResult::Ready(state) => *state,
        };
        self.dag_alpha_optimize_loop(
            input,
            config,
            engine,
            init,
            final_alpha_bound_only_enabled(),
            DagAlphaLoopResultUse::BoundsOnly,
            false,
        )
        .map(|artifact| artifact.output_bounds)
    }

    /// Shared optimization loop for DAG α-CROWN.
    ///
    /// Returns the optimized output bounds, final alpha state, and final sound
    /// reference map as one internal artifact.
    fn dag_alpha_optimize_loop(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        init: DagAlphaInitState,
        final_bound_only_gate: bool,
        result_use: DagAlphaLoopResultUse,
        retain_completed_on_deadline: bool,
    ) -> Result<DagAlphaCollectionArtifact> {
        let DagAlphaInitState {
            node_bounds,
            node_bounds_source,
            exec_order,
            output_dim,
            input_dim,
            relu_nodes,
            mut runtime,
            mut bilinear_alphas,
            mut bilinear_adam_m,
            mut bilinear_adam_v,
            mut mul_binary_alphas,
            mut mul_binary_adam_m,
            mut mul_binary_adam_v,
            has_bilinear,
            has_mul_binary,
            has_s_shaped,
            has_sqrt,
            has_reciprocal,
            invprop_enabled,
        } = init;

        let num_unstable = runtime.graph().num_unstable();
        let num_s_shaped = runtime.graph().monotone_alpha_names().count();
        let num_sqrt = runtime.graph().sqrt_alpha_names().count();
        let output_node_name = if self.output_node.is_empty() {
            exec_order.last().map(String::as_str)
        } else {
            Some(self.output_node.as_str())
        };
        // Patches identity seeds can take a patches fast-path before the Dense
        // output-seed augment. Until patches concretization carries the same
        // typed fold/proof provenance, such graphs are not treatment-eligible.
        let uses_patches_output_seed = output_node_name
            .and_then(|name| node_bounds.get(name))
            .is_some_and(|bounds| uses_patches_output_seed(self, bounds));
        // A gamma attempt is meaningful only when the output-seed fold can be
        // admitted. Keep this stricter than `invprop_enabled`: unsupported
        // disjunctions, empty/mismatched matrices, and margin-subset seeds must
        // not earn treatment telemetry for an algebraic no-op.
        let invprop_gamma_treatment_eligible =
            invprop_output_seed_treatment_eligible(config, output_dim) && !uses_patches_output_seed;
        // Only the ON treatment needs proof provenance and a CPU output-seed
        // fold. The OFF control keeps exact-zero gamma, so it is byte-equivalent
        // to identity and may retain the historical GPU warmup/suffix routes.
        let invprop_gamma_optimization_active =
            invprop_enabled && config.invprop.optimize_gammas && invprop_gamma_treatment_eligible;
        let no_alpha_optimizer = num_unstable == 0
            && !has_bilinear
            && !has_mul_binary
            && !has_s_shaped
            && !has_sqrt
            && !has_reciprocal;
        let gamma_only_optimizer = no_alpha_optimizer && invprop_gamma_optimization_active;
        let invprop_ascent_max_iters = if gamma_only_optimizer { 20 } else { 5 };
        debug!(
            "DAG α-CROWN: Starting optimization with {} unstable ReLU neurons across {} ReLU nodes, {} monotone S-shaped nodes, and {} sqrt nodes{}",
            num_unstable,
            relu_nodes.len(),
            num_s_shaped,
            num_sqrt,
            if invprop_enabled {
                " (INVPROP enabled)"
            } else {
                ""
            }
        );

        // A zero-iteration fixed-intermediate root collection means exactly
        // "initialize reusable alpha state, but do not optimize it." Return
        // the already-certified reference output directly instead of paying
        // the fixed-slope full-output CROWN baseline below.
        //
        // This distinction matters on deep image DAGs: the baseline is a
        // 100-row reverse walk (~40 s on CIFAR100_resnet_medium), while a
        // zero-iteration collection cannot consume that bound to update alpha.
        // With `fix_interm_bounds=true`, the collection caller returns the
        // reference node map and only keeps this initialized alpha state, so
        // the baseline is dead work. The explicit caller-local hint prevents
        // this optimization from changing the historical bounds-only
        // `iterations == 0` contract, which still consumes initial CROWN.
        if config.iterations == 0
            && config.fix_interm_bounds
            && config.skip_zero_iteration_collection_initial_bound
            && matches!(result_use, DagAlphaLoopResultUse::BoundsAndState)
        {
            let output_node = if self.output_node.is_empty() {
                exec_order.last().ok_or_else(|| {
                    NyError::InvalidSpec(
                        "DAG alpha zero-iteration request has no output node".to_string(),
                    )
                })?
            } else {
                &self.output_node
            };
            let output_bounds = node_bounds.get(output_node).cloned().ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "DAG alpha zero-iteration request missing output bounds for '{output_node}'"
                ))
            })?;
            crate::phase_telemetry::phase_marker(
                "dag-alpha-warmup skip root-collection configured-iterations=0",
            );
            return Ok(DagAlphaCollectionArtifact {
                output_bounds,
                alpha_state: runtime.into_graph_alpha_state(),
                reference_bounds: node_bounds,
                reference_bounds_source: node_bounds_source,
                completed_iterations: 0,
                optimizer_updates_completed: 0,
            });
        }

        // Adaptive skip: check if network is too deep for α-CROWN to help
        if config.adaptive_skip && relu_nodes.len() > config.adaptive_skip_depth_threshold {
            info!(
                "DAG α-CROWN: Adaptive skip triggered - {} ReLU nodes > threshold {}. \
                 For deep networks, bounds are often fundamentally loose and α-CROWN optimization \
                 provides no benefit. Falling back to CROWN.",
                relu_nodes.len(),
                config.adaptive_skip_depth_threshold
            );
            let alpha_state = runtime.into_graph_alpha_state();
            let output_bounds = if node_bounds_source.is_typed_cgan() {
                self.propagate_crown_with_engine_and_deadline_and_node_bounds(
                    input,
                    engine,
                    config.deadline,
                    Some(&node_bounds),
                )?
                .bounds
            } else {
                self.propagate_crown_with_engine_and_deadline(input, engine, config.deadline)?
                    .bounds
            };
            return Ok(DagAlphaCollectionArtifact {
                output_bounds,
                alpha_state,
                reference_bounds: node_bounds,
                reference_bounds_source: node_bounds_source,
                completed_iterations: 0,
                optimizer_updates_completed: 0,
            });
        }

        if tracing::enabled!(tracing::Level::DEBUG) {
            self.log_pre_loop_diagnostics(&exec_order, &node_bounds, &relu_nodes, input)?;
        }

        // #alpha-steering-proposal: resolve the dedicated α-gradient proposal
        // channel ONLY when the margin-gradient lane is armed with a spec
        // objective (the same condition under which the seam could consult
        // it), so unarmed runs never materialize an adapter. The owned `Arc`
        // outlives `ctx`; the context carries a borrow, never ownership, and
        // NEVER aliases `engine`.
        let (root_alpha_margin, spec_ascent) = root_alpha_margin_state(config);
        let alpha_steering_owned: Option<std::sync::Arc<dyn GemmEngine>> =
            if root_alpha_margin_gradient_enabled(root_alpha_margin) && spec_ascent.is_some() {
                crate::alpha_gradient_steering::steering_engine()
            } else {
                None
            };
        let ctx = DagAlphaLoopContext {
            input,
            exec_order: &exec_order,
            output_dim,
            input_dim,
            config,
            engine,
            alpha_steering: alpha_steering_owned.as_deref(),
            relu_nodes: &relu_nodes,
            has_bilinear,
            has_mul_binary,
        };

        // #root-alpha-gpu (A): build the warmup segment skeleton ONCE per loop
        // so every per-iteration warmup fold (bound + gradients) re-bakes only
        // the per-domain slots — static `Arc` weight payloads stay shared
        // across iterations instead of being re-materialized each extraction.
        // Dark behind NY_ROOT_ALPHA_GPU=1 (default OFF ⇒ the field stays
        // `None` and every warmup site takes the legacy extraction,
        // byte-identically). A build refusal also leaves `None` — fail closed.
        // `allow_pure_chain=false` matches the warmup extraction sites exactly.
        if root_alpha_gpu_enabled() && extract_skeleton_enabled() {
            let output_node_name = if self.output_node.is_empty() {
                exec_order.last().cloned()
            } else {
                Some(self.output_node.clone())
            };
            if let Some(output_node) = output_node_name {
                let skeleton = build_resnet_segment_skeleton(
                    self,
                    input,
                    &output_node,
                    &node_bounds,
                    &node_bounds,
                    Some(runtime.graph()),
                    /*allow_pure_chain=*/ false,
                );
                runtime.set_warmup_skeleton(skeleton);
            }
        }

        // #w4-gpu-dag-backward: the pre-loop initial CROWN bound below was measured
        // (sample profile, cifar100 CIFAR100_resnet_medium) at ~40s CPU wall — the
        // ENTIRE warmup budget, so alpha finished 0/20 iterations. Route it through
        // the SOUND GPU-resident resnet backward (identity seed, CROWN-initialized
        // alpha folded — the same certified enclosure the in-loop warmup bound uses)
        // when the suffix decomposes; on any refusal the proven CPU path below runs
        // unchanged. The expected output shape is captured BEFORE `node_bounds`
        // moves into `reference_bounds` so the flat GPU bound can be reshaped to
        // match the CPU path's output layout.
        let atomic_preloop_cuda_rows = self.atomic_preloop_cuda_rows(&ctx, &node_bounds, &runtime);
        let (gpu_initial_bounds, atomic_reference_retained, atomic_cuda_committed) =
            match atomic_preloop_cuda_rows {
                PreloopCudaRowsOutcome::NotRequested => (
                    self.try_gpu_warmup_bound(&ctx, &node_bounds, &runtime),
                    None,
                    false,
                ),
                PreloopCudaRowsOutcome::RefusedBeforeCommit { refusal } => {
                    info!(
                        reason = refusal.telemetry_reason(),
                        ?refusal,
                        "DAG alpha pre-loop CUDA-row transaction declined before backend \
                         commitment; preserving legacy initial-bound route"
                    );
                    (
                        self.try_gpu_warmup_bound(&ctx, &node_bounds, &runtime),
                        None,
                        false,
                    )
                }
                PreloopCudaRowsOutcome::CudaIntersection(bounds) => (Some(bounds), None, true),
                PreloopCudaRowsOutcome::ReferenceRetained { bounds, refusal } => {
                    info!(
                        reason = refusal.telemetry_reason(),
                        ?refusal,
                        "DAG alpha pre-loop CUDA-row transaction refused; retaining \
                         certified reference output without legacy fallback"
                    );
                    (None, Some(bounds), false)
                }
                PreloopCudaRowsOutcome::DeadlineExceeded => {
                    return Err(NyError::DeadlineExceeded(
                        "DAG alpha pre-loop atomic CUDA-row deadline exceeded".to_string(),
                    ));
                }
            };
        let output_shape: Option<Vec<usize>> = gpu_initial_bounds.as_ref().and_then(|_| {
            let output_node_name = if self.output_node.is_empty() {
                exec_order.last()?
            } else {
                &self.output_node
            };
            node_bounds
                .get(output_node_name)
                .map(|b| b.shape().to_vec())
        });

        let mut reference_bounds = super::reference_bounds::GraphAlphaReferenceBounds::new(
            node_bounds,
            self.graph_alpha_reference_bound_targets()?,
        )?;

        // Step 3: Optimization loop
        // Track element-wise best bounds across iterations:
        // - best_lower: maximum lower bound seen for each output dimension
        // - best_upper: minimum upper bound seen for each output dimension
        // Initialize from CROWN bounds to ensure α-CROWN never returns worse bounds.
        let crown_bounds = if let Some(bounds) = atomic_reference_retained {
            bounds
        } else {
            match gpu_initial_bounds {
                Some(bounds) => {
                    if atomic_cuda_committed {
                        info!("DAG α-CROWN: initial bound via atomic sound CUDA-row intersection");
                    } else {
                        info!(
                        "DAG α-CROWN: initial CROWN bound via sound GPU-resident resnet backward \
                         (#w4-gpu-dag-backward)"
                    );
                    }
                    match output_shape {
                        Some(ref shape) if shape.as_slice() != bounds.shape() => {
                            bounds.reshape(shape).unwrap_or_else(|_| bounds.clone())
                        }
                        _ => bounds,
                    }
                }
                None => {
                    // #dedup-root-collections Fix B: the CROWN backward pass used
                    // to re-run its internal Step-1 intermediate collection over
                    // the BIT-IDENTICAL root box that init just collected
                    // (measured ~73 s of duplicate work per root episode on
                    // vggnet16_2022 spec1; at tight budgets the recollection's
                    // deadline gate then discarded everything for vacuous IBP).
                    // Reuse the init map, but ONLY when it came from the same
                    // collector Step 1 would run (or a strictly tighter one), so
                    // no graph family gets weaker initial bounds than before:
                    //   - CrownIbp: same collector as Step 1's per-node CROWN-IBP
                    //     (or tighter than its >threshold IBP fallback). Only a
                    //     conv DAG would route Step 1 to forward-linear instead —
                    //     incomparable, so keep legacy behavior there.
                    //   - ForwardLinear: may come from either a conv DAG or the
                    //     dark sequential ConvTranspose reference lane. Reuse it
                    //     only when Step 1 independently selects forward-linear;
                    //     otherwise its collector is incomparable.
                    //   - CganSparseTargetComplete: begins with that same
                    //     complete forward-linear map and can only shrink it
                    //     through an atomic target plus downstream intersections.
                    //     It is compatible under the same Step-1 gate and lets
                    //     this bound consume the paid-for root tightening.
                    //   - Ibp: Step 1 might upgrade to per-node CROWN-IBP (and
                    //     its plain-IBP arm is the scalar f64 path, #4219), so
                    //     preserve the legacy internal collection byte-for-byte.
                    let step1_would_use_forward_linear =
                        self.should_collect_forward_linear_intermediate_reference();
                    let reuse_init_bounds = can_reuse_initial_node_bounds(
                        node_bounds_source,
                        step1_would_use_forward_linear,
                    );
                    let collector_cap = if reuse_init_bounds {
                        None
                    } else {
                        super::bounds::crown_ibp_collector_cap()
                    };
                    match collector_cap {
                        Some(cap) => {
                            eprintln!(
                                "[NY_CROWN_IBP_COLLECTOR_CAP_V1] stage=alpha-preloop-dispatch \
                             requested_secs={} reuse_init_bounds=false",
                                cap.as_secs(),
                            );
                            self.propagate_crown_with_engine_and_deadline_and_node_bounds_and_crown_ibp_cap(
                            input,
                            engine,
                            config.deadline,
                            None,
                            Some(cap),
                        )?
                        .bounds
                        }
                        None => {
                            self.propagate_crown_with_engine_and_deadline_and_node_bounds(
                                input,
                                engine,
                                config.deadline,
                                reuse_init_bounds.then(|| reference_bounds.current()),
                            )?
                            .bounds
                        }
                    }
                }
            }
        };
        let mut best_lower: ArrayD<f32> = crown_bounds.lower().clone();
        let mut best_upper: ArrayD<f32> = crown_bounds.upper().clone();
        // Use finite-only sum to prevent -Inf poisoning the early-stopping metric (#2857).
        // Prior layout-agnostic fix: #1939.
        let mut best_lower_sum: f32 = finite_lower_sum(crown_bounds.lower());
        let mut prev_best_lower_sum = best_lower_sum;
        let mut best_invprop_gap: Option<f64> = None;
        let mut prev_best_invprop_gap: Option<f64> = None;
        // #root-alpha-margin (typed default OFF with env override): armed only when
        // the effective gate is set AND the caller supplied a multi-row spec
        // objective. When either is absent this stays `None` and every branch below
        // is inert, so the loop is byte-identical to the legacy path.
        let margin_gradient_requested = root_alpha_margin_gradient_enabled(root_alpha_margin);
        let margin_hinge_steering = root_alpha_margin_hinge_enabled(margin_gradient_requested);
        // A GraphAlphaState snapshot is the complete optimizer state only for
        // the ReLU-only path.  Refuse rather than mix a best ReLU checkpoint
        // with later bilinear, monotone, sqrt, reciprocal, or INVPROP state.
        // AnalyticChain is required because the bounded GPU adjoint below is
        // the only objective-directed gradient implementation in this lane.
        let (multiobj_joint_alpha_armed, root_alpha_true_armed) = if margin_gradient_requested {
            (
                multiobj_joint_alpha_enabled(),
                gradients::root_alpha_true_enabled(),
            )
        } else {
            (false, false)
        };
        let margin_gradient_eligibility = margin_gradient_eligibility(
            margin_gradient_requested,
            spec_ascent.is_some(),
            matches!(
                config.gradient_method,
                crate::bounds::GradientMethod::AnalyticChain
            ),
            !has_bilinear
                && !has_mul_binary
                && !has_s_shaped
                && !has_sqrt
                && !has_reciprocal
                && !invprop_gamma_optimization_active,
            multiobj_joint_alpha_armed,
            root_alpha_true_armed,
        );
        let margin_gradient_eligible = margin_gradient_eligibility.is_eligible();
        // Null-run guard: this repo has repeatedly lost sessions to gates that were
        // armed in the environment but inert in the code (quarantine-dead readers,
        // configs that never reached the operative lane). Say plainly, once, whether
        // this run is ranking or on the legacy path, so a measurement can never be
        // ambiguous about which arm it actually exercised.
        if root_alpha_margin {
            info!(
                "DAG α-CROWN #root-alpha-margin: gate ARMED, spec objective {} — {}",
                if spec_ascent.is_some() {
                    "PRESENT"
                } else {
                    "ABSENT"
                },
                if spec_ascent.is_some() {
                    "ranking α iterates by margin hinge"
                } else {
                    "INERT, falling back to the legacy last-iterate α"
                },
            );
        }
        if margin_gradient_requested {
            if margin_gradient_eligible {
                info!(
                    "DAG α-CROWN #root-alpha-margin-gradient: gate ARMED — \
                     ELIGIBLE, resident joint admission pending per iteration"
                );
            } else {
                info!(
                    "DAG α-CROWN #root-alpha-margin-gradient: gate ARMED — \
                     INERT reason={}, preserving the pre-existing alpha-gradient policy exactly",
                    margin_gradient_eligibility.reason()
                );
            }
        }
        let mut best_margin_score: Option<f32> = None;
        let mut prev_best_margin_score: Option<f32> = None;
        let mut best_margin_alpha: Option<GraphAlphaState> = None;
        let mut best_margin_iter: usize = 0;
        // Objective-specific convergence is authorized only after the previous
        // update actually came from the resident joint adjoint. Gate
        // eligibility alone is not a dispatch result.
        let mut previous_margin_joint_dispatched = false;
        let mut no_improve_iters = 0usize;
        // #alpha-zero-yield state: when the ascent last actually improved, and
        // the size of the window it is being paid out of.
        let mut last_improvement_at = std::time::Instant::now();
        let ascent_window = config
            .deadline
            .map(|d| d.saturating_duration_since(std::time::Instant::now()));
        let mut lr = config.learning_rate;
        let mut infeasible_bounds: Option<BoundedTensor> = None;
        let mut total_gradient_skips: usize = 0;

        // The `Analytic` method takes its per-ReLU gradients directly from the CPU
        // backward's in-place fill; replacing that backward with the GPU warmup
        // bound would leave them zero (sound, but alpha would never move). Only
        // methods with their own gradient source (AnalyticChain via the GPU
        // warmup-gradient hook / SPSA / FiniteDifferences via bound evals) may take
        // the in-loop GPU bound. The PRE-LOOP initial bound above fills no
        // gradients, so it is exempt from this guard.
        // The GPU-resident warmup bound bypasses the CPU INVPROP seed augment,
        // so an active gamma optimizer must use the CPU backward that owns the
        // fold/proof provenance. The OFF control keeps an exact-zero identity
        // seed and therefore retains the historical GPU route.
        let in_loop_gpu_bound_ok = !(matches!(
            config.gradient_method,
            crate::bounds::GradientMethod::Analytic
        ) || invprop_gamma_optimization_active);
        // A terminal optimizer update is dead only when the caller discards
        // the state. Root collection returns it for BaB warm-starting and
        // immediately re-evaluates it, so that path must retain the legacy
        // final gradient/update even when the experiment gate is enabled.
        let final_bound_only = result_use.terminal_bound_only(final_bound_only_gate);
        if final_bound_only_gate && !final_bound_only {
            debug!(
                ?result_use,
                "DAG α-CROWN: NY_ALPHA_FINAL_BOUND_ONLY fail-closed because returned state is observable"
            );
        }

        // #phase-telemetry (dark, NY_PHASE_TELEMETRY=1, print-only): phase
        // markers for the dag-alpha warmup loop — lever pricing needs phase
        // boundaries, not single-row wall deltas (~±15% layout noise across
        // builds). The gate is checked BEFORE every `format!` so the
        // default-unset path stays allocation-free and byte-identical.
        // Track both begun and publication-safe folds.  They differ when a
        // child backend refuses or returns late: only the latter may authorize
        // a phase-cap checkpoint.
        if crate::phase_telemetry::phase_telemetry_enabled() {
            crate::phase_telemetry::phase_marker(&format!(
                "dag-alpha-warmup loop-enter planned-iters={}",
                config.iterations
            ));
        }
        let mut phase_iters_started = 0usize;
        let mut phase_iters_completed = 0usize;
        let mut phase_optimizer_updates_completed = 0usize;
        // #wall-refresh-cumulative: the configured fraction is one TOTAL
        // alpha-loop refresh pool, not a fresh geometric slice on every
        // improving iteration. The old `0.25 * remaining` recurrence consumed
        // 1-(0.75^N) of the root window (76% after five refreshes), starving
        // BaB. Every completed candidate remains a certified enclosure, and
        // exhaustion simply keeps the previous sound reference map. A run
        // without a global deadline and without a configured maximum preserves
        // the historical unbounded collection path.
        let alpha_refresh_fraction = alpha_refresh_fraction(config);
        let alpha_refresh_max_duration = config
            .reference_refresh_max_secs
            .map(std::time::Duration::from_secs);
        let mut alpha_refresh_budget_remaining: Option<std::time::Duration> = None;
        'alpha_iterations: for iter in 0..config.iterations {
            phase_iters_started = iter + 1;
            if crate::phase_telemetry::phase_telemetry_enabled() {
                crate::phase_telemetry::phase_marker(&format!(
                    "dag-alpha-warmup iter={iter} start"
                ));
            }
            // Deadline check (#2962): bail early if verification timeout budget
            // is exhausted. Return current best bounds instead of running all iterations.
            // Matches pattern in alpha_crown_loop.rs:112 and bounds/alpha.rs:114.
            if config.past_deadline() {
                info!(
                    "DAG α-CROWN: deadline exceeded at iteration {}/{}, returning best bounds",
                    iter, config.iterations
                );
                break;
            }
            let is_last_iter = iter == config.iterations - 1;
            let need_grad =
                alpha_iteration_needs_gradient(iter, config.iterations, final_bound_only);
            let node_bounds = reference_bounds.current();

            // Initialize gradients for each ReLU node
            let mut gradients: Vec<Array1<f32>> = if need_grad {
                relu_nodes
                    .iter()
                    .map(|(name, _)| {
                        let pre_act = self.relu_preactivation_bounds(
                            name,
                            input,
                            node_bounds,
                            "dag-alpha-gradient-init",
                        )?;
                        Ok(Array1::zeros(pre_act.len()))
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            // Separate upper-path gradient buffer (#3393).
            let mut gradients_upper: Vec<Array1<f32>> =
                gradients.iter().map(|g| Array1::zeros(g.len())).collect();

            // Run backward pass through DAG with alpha values.
            // Pass bilinear/mul_binary alphas so nonlinear nodes use interpolated bounds (#3287, #3439).
            let (bilinear_ref, mul_binary_ref) =
                gradients::alpha_refs(&ctx, &bilinear_alphas, &mul_binary_alphas);
            // A nonzero INVPROP dual conditions this backward pass on the
            // assume-violation region. Such a bound may prove that region
            // infeasible, but while it remains feasible it is not a global
            // output-box enclosure and must never enter the elementwise best
            // box returned by this API. Zero gamma is the exact identity fold
            // and remains eligible for ordinary alpha-CROWN retention.
            let returnable_box_iterate = runtime.invprop().is_none_or(|state| {
                state
                    .all_ny_params()
                    .iter()
                    .all(|gamma| gamma.to_bits() == 0.0_f32.to_bits())
            });
            // #root-alpha-gpu (B): this iteration's loop-top GPU fold cache
            // (bound + local-rule gradients + segments from ONE kernel call).
            // Declared per-iteration so a previous iteration's fold can never
            // leak; consumed by the gradient site below. The reuse is valid
            // because α only changes at the END of the iteration
            // (`update_all_alphas` below), so the loop-top fold's α is still
            // current at the gradient site.
            let mut warmup_iter_cache: Option<gradients::WarmupGpuIterCache> = None;

            // #unsat-keystone: GPU-resident warmup BOUND fast-path (the #1 wall — the CPU
            // dag_alpha_backward_pass is ~7s/iter on cifar100 → warmup eats the whole
            // budget → 0 BaB domains). When it fires, gradients stay zero-initialized here
            // and are filled by the GPU warmup-gradient path at the gradient site below
            // (for a fully-decomposed resnet suffix that covers every ReLU). Gated, sound,
            // CPU fallback. Non-soundness-critical (warmup alpha).
            let evaluated_fold_scope = invprop_gamma_optimization_active
                .then(crate::execution_telemetry::begin_invprop_evaluated_fold_scope);
            let (
                mut concrete_bounds,
                certified_finite_inversion,
                invprop_gap_score,
                invprop_row_gap_scores,
                completed_conditioned_fold,
            ) = {
                match in_loop_gpu_bound_ok
                    .then(|| {
                        if !need_grad {
                            self.try_gpu_warmup_bound_only(&ctx, node_bounds, &runtime)
                        } else if retain_warmup_iter_cache(
                            root_alpha_gpu_enabled(),
                            margin_gradient_eligible,
                        ) {
                            // #root-alpha-gpu (B): take the FULL fold once and keep
                            // its cache for the gradient site (one GPU fold per
                            // iteration instead of two). The margin child also
                            // retains this cache so a refused joint adjoint can use
                            // the same resident local-rule gradients as its
                            // child-disabled twin. Bound value is identical to the
                            // wrapper below by construction.
                            self.try_gpu_warmup_bound_full(&ctx, node_bounds, &runtime)
                                .map(|(bounds, mut cache)| {
                                    cache.iter = iter;
                                    warmup_iter_cache = Some(cache);
                                    bounds
                                })
                        } else {
                            self.try_gpu_warmup_bound(&ctx, node_bounds, &runtime)
                        }
                    })
                    .flatten()
                {
                    Some(bounds) => (bounds, false, None, Vec::new(), true),
                    None => {
                        let fold = if need_grad {
                            self.dag_alpha_backward_pass_with_engine_and_infeasibility(
                                input,
                                node_bounds,
                                &exec_order,
                                output_dim,
                                input_dim,
                                runtime.relu_name_to_idx(),
                                runtime.graph(),
                                if invprop_gamma_optimization_active {
                                    runtime.invprop()
                                } else {
                                    None
                                },
                                &mut gradients,
                                &mut gradients_upper,
                                engine,
                                bilinear_ref,
                                mul_binary_ref,
                                config.deadline,
                            )
                        } else {
                            self.dag_alpha_bound_pass_with_engine_and_infeasibility(
                                input,
                                node_bounds,
                                &exec_order,
                                output_dim,
                                input_dim,
                                runtime.relu_name_to_idx(),
                                runtime.graph(),
                                if invprop_gamma_optimization_active {
                                    runtime.invprop()
                                } else {
                                    None
                                },
                                engine,
                                bilinear_ref,
                                mul_binary_ref,
                                config.deadline,
                            )
                        };
                        match fold {
                            Ok(fold) => fold,
                            Err(error)
                                if retain_completed_deadline_error(
                                    retain_completed_on_deadline,
                                    phase_iters_completed,
                                    &error,
                                ) =>
                            {
                                info!(
                                    completed_iterations = phase_iters_completed,
                                    "DAG α-CROWN: retaining completed artifact after a later bound fold reached the phase deadline"
                                );
                                break 'alpha_iterations;
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
            };
            if completed_conditioned_fold {
                if let Some(scope) = evaluated_fold_scope {
                    scope.commit();
                }
            }

            if let Some(state) = runtime.invprop_mut() {
                if certified_finite_inversion {
                    state.mark_infeasible(0)?;
                    state.apply_infeasible_mask(&mut concrete_bounds);
                    infeasible_bounds = Some(concrete_bounds);
                    phase_iters_completed = iter + 1;
                    break;
                }
            }

            // Update element-wise best bounds with layout-agnostic iteration.
            // This handles non-standard layout arrays and shape-only ndim mismatch
            // as long as element counts match (#2076, #2087).
            // Skip during warmup window to avoid locking in noisy early-iteration bounds.
            // Matches α,β-CROWN's start_save_best (optimized_bounds.py:785-797).
            if returnable_box_iterate && config.should_save_best(iter, is_last_iter) {
                update_elementwise_best_bounds(
                    &mut best_lower,
                    &mut best_upper,
                    &concrete_bounds,
                    iter,
                )?;
            }

            // #root-alpha-margin (default OFF): rank this iterate by the SPEC
            // objective and remember the α that produced the best one.
            //
            // The legacy loop returns its LAST α iterate, which is one optimizer
            // update ahead of the last bound it evaluated, and there is no best-α
            // snapshot anywhere. On multi-class properties the ascent objective is a
            // sum over RAW output dims while the property is a conjunction of MARGIN
            // rows, so those iterations can hand the downstream spec pass an α that is
            // strictly worse for the margins than the one the loop started with.
            //
            // SOUNDNESS: selection only. The score never decides a verdict and never
            // feeds a bound — it only picks WHICH α to keep, and every α ∈ [0,1] is
            // valid. Worst case is a weaker bound, never a wrong one.
            let mut margin_gradient_objective: Option<BindingMarginObjective> = None;
            if let Some(ascent) = spec_ascent {
                // Layout-agnostic (#1939, #2076, #2087): this loop genuinely sees
                // non-standard layouts, where `as_slice()` returns None. Falling back
                // to a logical-order copy keeps the gate from silently becoming a
                // no-op on exactly those graphs — a null run is worse than a cost,
                // and one Vec per iteration is noise next to a backward pass.
                let as_vec = |arr: &ArrayD<f32>| -> Vec<f32> {
                    arr.as_slice()
                        .map_or_else(|| arr.iter().copied().collect(), <[f32]>::to_vec)
                };
                let lo = as_vec(concrete_bounds.lower());
                let hi = as_vec(concrete_bounds.upper());
                // #spec-axis-alpha slice 2d: install δ slots for the K WORST
                // margins at the first scored iterate (δ = 0 ⇒ this iteration
                // and the next fold are bit-identical to shared α — the
                // #iter0-alpha-parity anchor extends through installation).
                // Carrier rows map to ORIGINAL output rows through the same
                // deterministic subset helper the seed used.
                if iter == 0
                    && config.alpha_spec_slots > 0
                    && runtime.graph().spec_slot_rows.is_empty()
                {
                    let subset = crate::output_margin_seed::margin_subset_indices(output_dim);
                    let installed = spec_axis::assign_spec_slots(
                        runtime.graph_mut(),
                        &lo,
                        subset.as_deref(),
                        config.alpha_spec_slots,
                    );
                    if installed > 0 {
                        info!(
                            "DAG α-CROWN #spec-axis-alpha iter 0: installed {installed} δ \
                             slot(s) for worst margins {:?}",
                            runtime.graph().spec_slot_rows
                        );
                    }
                }
                {
                    let (lo, hi) = (lo.as_slice(), hi.as_slice());
                    if let Some(score) = ascent.hinge_score(lo, hi) {
                        let improved = best_margin_score.is_none_or(|best| score > best);
                        if improved {
                            best_margin_score = Some(score);
                            best_margin_alpha = Some(runtime.snapshot_graph());
                            best_margin_iter = iter;
                        }
                        debug!(
                            "DAG α-CROWN #root-alpha-margin iter {iter}: hinge={score:.6} \
                             best={:.6} (iter {best_margin_iter}) rows_verified={}/{}",
                            best_margin_score.unwrap_or(score),
                            ascent.verified_rows(lo, hi),
                            ascent.rows.len(),
                        );
                    }
                    if margin_gradient_eligible {
                        // #root-alpha-margin-hinge: steer by the hinge subgradient
                        // (sum of all violated rows) so the direction agrees with
                        // the hinge score selection is judged by. Default off ⇒
                        // the single-worst-row path below, byte-identical.
                        margin_gradient_objective = if margin_hinge_steering {
                            hinge_margin_lower_objective(ascent, lo, hi)
                        } else {
                            binding_margin_lower_objective(ascent, lo, hi)
                        };
                        if let Some(binding) = margin_gradient_objective.as_ref() {
                            info!(
                                "DAG α-CROWN #root-alpha-margin-gradient iter {iter}: \
                                 binding_row={} slack={:.6}",
                                binding.row_index, binding.slack
                            );
                        } else {
                            info!(
                                "DAG α-CROWN #root-alpha-margin-gradient iter {iter}: \
                                 no finite unverified binding row; bounded local fallback pending"
                            );
                        }
                    }
                }
            }

            // Finite-only sum for early stopping (#2857). Layout-agnostic (#1939).
            let lower_sum: f32 = finite_lower_sum(concrete_bounds.lower());
            let gamma_gap_control =
                invprop_gamma_optimization_active && invprop_gap_score.is_some();
            if let Some(gap) = invprop_gap_score {
                if best_invprop_gap.is_none_or(|best| gap > best) {
                    best_invprop_gap = Some(gap);
                }
            }

            // #iter0-alpha-parity (dark, NY_ITER0_PARITY_TRACE=1, print-only):
            // emit the evaluated bound IMMEDIATELY after the fold, before any
            // gradient/update/deadline exit below can suppress the end-of-
            // iteration logs. The iteration-0 line is the loop-fold number the
            // parity investigation compares against the pre-loop CROWN
            // baseline (best_lower_sum at iter 0 IS that baseline).
            if crate::iter0_parity_trace::iter0_parity_trace_enabled() {
                eprintln!(
                    "[iter0-parity] alpha-iter iter={iter} lower_sum={lower_sum:.6e} \
                     baseline_best_lower_sum={best_lower_sum:.6e}"
                );
            }

            // NaN detection: if any bound element is NaN, the backward pass produced
            // garbage. Break early to avoid wasting remaining iterations — the
            // post-loop has_nan check will fall back to CROWN. (#2597)
            if concrete_bounds.lower().iter().any(|v| v.is_nan())
                || concrete_bounds.upper().iter().any(|v| v.is_nan())
            {
                warn!(
                    "DAG α-CROWN: NaN in bounds at iteration {iter}, aborting optimization (#2597)"
                );
                break;
            }
            // The bound fold is now an atomic, finite publication.  Later
            // gradient/update/reference work may stop, but it cannot erase
            // this completed checkpoint.
            phase_iters_completed = iter + 1;

            // #invprop-alpha-budget: mid-iteration deadline check. The loop-top
            // check cannot see a backward pass that overran the budget INSIDE
            // this iteration; without this the iteration would go on to spend
            // more past-deadline time on gradient probes and the reference-
            // bound refresh before the next loop-top check fires. Sound: only
            // stops optimizing sooner, returning the elementwise best bounds.
            if config.past_deadline() {
                if returnable_box_iterate && !config.should_save_best(iter, is_last_iter) {
                    update_elementwise_best_bounds(
                        &mut best_lower,
                        &mut best_upper,
                        &concrete_bounds,
                        iter,
                    )?;
                }
                info!(
                    "DAG α-CROWN: deadline exceeded after iteration {}/{} backward pass, \
                     returning best bounds",
                    iter, config.iterations
                );
                break;
            }

            // Spec-proven early-exit (#warmup-early-exit). When the single-objective
            // warmup carries a `spec_early_exit`, project the elementwise BEST output
            // bounds (the bounds this loop will actually return) onto the objective and
            // stop the moment they already prove the property against the threshold.
            // SOUND: this only stops optimizing sooner — the projected bound at the exit
            // iteration is a valid over-approximation that clears the threshold; no bound
            // *computation* changes, and `None` callers (every non-warmup caller) skip
            // this entirely.
            if let Some(spec) = config.spec_early_exit.as_ref() {
                if let (Some(lo), Some(hi)) = (best_lower.as_slice(), best_upper.as_slice()) {
                    if let Some((root_lower, root_upper)) = spec.project_bounds(lo, hi) {
                        if spec.is_verified(root_lower, root_upper) {
                            debug!(
                                "DAG α-CROWN: spec-proven early-exit at iter {} \
                                 (root bound [{:.4}, {:.4}] clears threshold {})",
                                iter, root_lower, root_upper, spec.threshold
                            );
                            break;
                        }
                    }
                }
            }

            // Track best lower_sum for early stopping
            let improved_output = lower_sum > best_lower_sum;
            if improved_output {
                best_lower_sum = lower_sum;
            }

            // INVPROP: projected gamma ascent on the output-seed duals (DAG path,
            // Stage 3). Gated on optimize_gammas (default OFF => normal runs untouched).
            // Perturbs only the output-seed group; research per-layer gammas are
            // unrelated to this objective and must not make a seed write appear
            // effective. Soundness is gamma-independent: feasible conditioned
            // bounds never enter the global best box, and only typed finite
            // inversion can prove emptiness. A bounded deterministic one-sided
            // SPSA estimate steers the search; ordinary probe bounds are discarded,
            // while a typed inversion is transactionally promoted.
            // THROUGHPUT GUARD (see alpha_crown_loop): each probe is one extra
            // backward — expensive on conv-resnets — so cap it to the first few
            // iters and skip near the deadline. A post-probe deadline check below
            // prevents later alpha/supplement/reference work after a probe consumes
            // the remaining budget.
            let mut gamma_probe_deadline_exceeded = false;
            let gamma_probe_available = invprop_gamma_optimization_active
                && need_grad
                && iter < invprop_ascent_max_iters
                && !config.past_deadline()
                && invprop_gap_score.is_some();
            if gamma_probe_available {
                let seed_key = output_node_name.unwrap_or_default();
                let base_params = runtime
                    .invprop()
                    .and_then(|state| state.layer_gammas(seed_key))
                    .filter(|gammas| gammas.active)
                    .map(|gammas| gammas.gammas.clone());
                if let Some(base_params) = base_params.filter(|params| {
                    let (_, constraint_dim, neuron_dim) = params.dim();
                    !params.is_empty()
                        && params.dim().0 == 2
                        && constraint_dim
                            == config
                                .output_constraints
                                .as_ref()
                                .map_or(0, |constraints| constraints.num_constraints())
                        && (neuron_dim == 1 || neuron_dim == output_dim)
                        && params
                            .iter()
                            .all(|value| value.is_finite() && *value >= 0.0)
                }) {
                    crate::execution_telemetry::record_invprop_gamma_step_attempted();
                    let lr_g =
                        if config.invprop.gamma_lr.is_finite() && config.invprop.gamma_lr > 0.0 {
                            config.invprop.gamma_lr
                        } else {
                            0.5
                        };
                    let delta = 0.1f32;
                    let mut perturbed = base_params.clone();
                    for (i, value) in perturbed.iter_mut().enumerate() {
                        *value = (*value + delta * invprop_spsa_sign(iter, i)).max(0.0);
                    }
                    'gamma_transaction: {
                        if install_invprop_seed_params(&mut runtime, seed_key, &perturbed).is_err()
                        {
                            install_invprop_seed_params(&mut runtime, seed_key, &base_params)?;
                            break 'gamma_transaction;
                        }
                        let (blr, mbr) =
                            gradients::alpha_refs(&ctx, &bilinear_alphas, &mul_binary_alphas);
                        let evaluated_probe_scope =
                            crate::execution_telemetry::begin_invprop_evaluated_fold_scope();
                        let probe_result = self.dag_alpha_bound_pass_with_engine_and_infeasibility(
                            input,
                            node_bounds,
                            &exec_order,
                            output_dim,
                            input_dim,
                            runtime.relu_name_to_idx(),
                            runtime.graph(),
                            runtime.invprop(),
                            engine,
                            blr,
                            mbr,
                            config.deadline,
                        );
                        let (certified_infeasible, probe_row_gaps, completed_fold) =
                            match probe_result {
                                Ok((_bounds, certified, _gap, row_gaps, completed)) => {
                                    (certified, row_gaps, completed)
                                }
                                Err(error) => {
                                    install_invprop_seed_params(
                                        &mut runtime,
                                        seed_key,
                                        &base_params,
                                    )?;
                                    if matches!(error, NyError::DeadlineExceeded(_)) {
                                        gamma_probe_deadline_exceeded = true;
                                        break 'gamma_transaction;
                                    }
                                    if invprop_gamma_probe_can_noop(&error) {
                                        break 'gamma_transaction;
                                    }
                                    return Err(error);
                                }
                            };

                        let perturbed_changed = !params_bit_identical(&perturbed, &base_params);
                        if completed_fold && certified_infeasible && perturbed_changed {
                            crate::execution_telemetry::record_invprop_gamma_step_applied();
                            evaluated_probe_scope.commit();
                            let state = runtime.invprop_mut().ok_or_else(|| {
                                NyError::InternalError(
                                    "DAG INVPROP proof promotion lost state".to_string(),
                                )
                            })?;
                            state.mark_infeasible(0)?;
                            state.apply_infeasible_mask(&mut concrete_bounds);
                            infeasible_bounds = Some(concrete_bounds);
                            break 'alpha_iterations;
                        }
                        drop(evaluated_probe_scope);

                        // The probe is never authoritative state. Restore the
                        // exact evaluated base before inspecting its result so
                        // every refusal/non-finite branch is transactional.
                        install_invprop_seed_params(&mut runtime, seed_key, &base_params)?;
                        let Some(updated) = invprop_projected_spsa_update(
                            &base_params,
                            &invprop_row_gap_scores,
                            &probe_row_gaps,
                            f64::from(delta),
                            lr_g,
                            iter,
                        ) else {
                            break 'gamma_transaction;
                        };
                        if install_invprop_seed_params(&mut runtime, seed_key, &updated).is_err() {
                            install_invprop_seed_params(&mut runtime, seed_key, &base_params)?;
                            break 'gamma_transaction;
                        }
                        let write_effective = runtime
                            .invprop()
                            .and_then(|state| state.layer_gammas(seed_key))
                            .map(|gammas| params_bit_identical(&gammas.gammas, &updated))
                            .unwrap_or(false);
                        if !write_effective {
                            install_invprop_seed_params(&mut runtime, seed_key, &base_params)?;
                            break 'gamma_transaction;
                        }
                        crate::execution_telemetry::record_invprop_gamma_step_applied();
                    }
                }
            }

            if gamma_probe_deadline_exceeded {
                info!(
                    "DAG α-CROWN: optional INVPROP probe reached the deadline at iteration {iter}; returning best global bounds"
                );
                break 'alpha_iterations;
            }

            if config.past_deadline() {
                info!(
                    "DAG α-CROWN: deadline exceeded by INVPROP probe at iteration {iter}, returning best global bounds"
                );
                break;
            }

            // Early stopping must observe the objective this arm actually
            // optimizes. Counting raw-logit stagnation while the binding
            // margin improves would terminate the experiment for the same
            // wrong-objective reason this gate repairs.
            let best_improvement = if gamma_gap_control {
                match (best_invprop_gap, prev_best_invprop_gap) {
                    (Some(best), Some(previous)) => best - previous,
                    (Some(_), None) => f64::INFINITY,
                    _ => 0.0,
                }
            } else if previous_margin_joint_dispatched {
                match (best_margin_score, prev_best_margin_score) {
                    (Some(best), Some(previous)) => f64::from(best - previous),
                    (Some(_), None) => f64::INFINITY,
                    _ => 0.0,
                }
            } else {
                f64::from(best_lower_sum - prev_best_lower_sum)
            };
            // #alpha-divergence-bail (dark, `NY_ALPHA_DIVERGENCE_BAIL=1`, default
            // OFF ⇒ byte-identical): abandon an ascent that has visibly blown up
            // instead of paying `early_stop_patience` more iterations to discover
            // it stagnated.
            //
            // MEASURED (docs/ROOT_ALPHA_STEP_EXPLODES_AND_STALLS_2026-07-29.md,
            // CIFAR100_resnet_medium prop_idx_7500, wgpu/Metal): the CROWN
            // initializer gives `best_lower_sum = -1989.90` and α iteration 0
            // returns `-2.15e23` with `best_impr = 0.000e0` — twenty orders of
            // magnitude worse than its own starting point, reproducibly. Each
            // iteration costs ~207 s, and patience is 10, so the loop spends
            // ~2070 s confirming that a state which exploded on step one is not
            // going to recover. On a 100 s scored budget that is the entire run.
            //
            // An iterate this far below the best is not a slow-converging ascent;
            // the α state itself is diverging, and every subsequent iterate
            // continues from that state. Stop and keep the best.
            //
            // SOUND: pure early exit. The loop already returns the elementwise
            // BEST bounds seen, and this path takes the same `should_save_best`
            // route as ordinary convergence, so the returned enclosure is one the
            // unguarded loop would also have been entitled to return. Stopping
            // sooner can return a looser certified enclosure; it cannot
            // manufacture an invalid bound.
            let diverged = !gamma_gap_control && alpha_divergence_bail_enabled() && {
                let best = f64::from(best_lower_sum);
                let now = f64::from(lower_sum);
                !now.is_finite() || (best - now) > ALPHA_DIVERGENCE_FACTOR * best.abs().max(1.0)
            };
            if diverged {
                if returnable_box_iterate && !config.should_save_best(iter, false) {
                    update_elementwise_best_bounds(
                        &mut best_lower,
                        &mut best_upper,
                        &concrete_bounds,
                        iter,
                    )?;
                }
                warn!(
                    "DAG α-CROWN #alpha-divergence-bail: iterate at {lower_sum:.3e} is \
                     catastrophically below the best {best_lower_sum:.3e} at iteration \
                     {iter}; the α state has diverged. Abandoning the ascent and keeping \
                     the best iterate."
                );
                break;
            }

            if best_improvement < f64::from(config.tolerance) {
                no_improve_iters += 1;
            } else {
                no_improve_iters = 0;
                last_improvement_at = std::time::Instant::now();
            }
            // #alpha-zero-yield (shipped dark pending a retained,
            // promotion-grade A/B; typed presets and the legacy env seam
            // remain available for controlled runs).
            //
            // Invariant I3 of docs/DESIGN_MARGINAL_VALUE_SCHEDULER_2026-08-08.md:
            // stop paying for zero. `early_stop_patience` already implements
            // that idea, but it counts ITERATIONS -- which is the same
            // fixed-constant defect the design indicts, because an iteration is
            // not a fixed price. On cifar100_resnet_medium an iteration costs
            // ~4-5 s against a 40 s window, so a patience of 10 cannot fire
            // before the window closes: the measured ascent runs 7-10 iterations
            // at `best_impr = 0.000e0` and is cut off by the deadline having
            // improved on its own initialiser exactly zero times.
            //
            // Measured elsewhere in the campaign: cutting the ascent to a single
            // iteration leaves the displayed dense-head scalar at 624.0831 for
            // 1, 3 and 7 iterations and returns ~18 s to search. That scalar
            // does not establish elementwise or published-bound bit identity.
            //
            // So retire the phase on a fraction of ITS OWN WINDOW spent without
            // yield, which is budget-relative by construction (invariant I1).
            //
            // SOUND: pure early exit on the same `should_save_best` route as
            // ordinary convergence -- the loop returns the elementwise best
            // bounds it has already certified. Stopping sooner can return a
            // looser certified enclosure; it cannot manufacture an invalid
            // bound.
            if let Some(frac) = alpha_zero_yield_frac(config) {
                if let Some(window) = ascent_window {
                    let idle = last_improvement_at.elapsed();
                    if no_improve_iters >= 1 && idle > window.mul_f64(frac) {
                        if returnable_box_iterate && !config.should_save_best(iter, false) {
                            update_elementwise_best_bounds(
                                &mut best_lower,
                                &mut best_upper,
                                &concrete_bounds,
                                iter,
                            )?;
                        }
                        info!(
                            "DAG α-CROWN #alpha-zero-yield: {:.1}s of a {:.1}s window with no \
                             improvement over the best iterate (iteration {iter}); retiring the \
                             ascent and returning the budget.",
                            idle.as_secs_f64(),
                            window.as_secs_f64()
                        );
                        break;
                    }
                }
            }
            if iter > 0 && no_improve_iters >= config.early_stop_patience && !gamma_probe_available
            {
                if returnable_box_iterate && !config.should_save_best(iter, false) {
                    update_elementwise_best_bounds(
                        &mut best_lower,
                        &mut best_upper,
                        &concrete_bounds,
                        iter,
                    )?;
                }
                info!(
                    "DAG α-CROWN: Converged at iteration {} (best improvement < {} for {} iters)",
                    iter, config.tolerance, no_improve_iters
                );
                break;
            }

            // Pilot iteration check: after SECOND iteration, verify α-CROWN helps.
            // Must be iter >= 1 (not iter == 0) because iteration 0 uses CROWN-initialized
            // alpha values and always produces bounds identical to plain CROWN. The first
            // alpha update happens at the END of iter 0, so iter 1 is the first iteration
            // that reflects optimized alpha values. (#3293)
            if iter == 1 && config.adaptive_skip && config.adaptive_skip_pilot && !gamma_gap_control
            {
                // Compute improvement over initial CROWN bounds (#2857, #1939).
                let initial_lower_sum: f32 = finite_lower_sum(crown_bounds.lower());
                let pilot_improvement = if previous_margin_joint_dispatched {
                    // The first finite margin score is the pilot baseline. The
                    // regular patience metric above remains authoritative for
                    // later iterations; declining this legacy raw-output pilot
                    // avoids an objective-crossing early exit.
                    best_margin_score
                        .zip(prev_best_margin_score)
                        .map_or(f32::INFINITY, |(best, previous)| best - previous)
                } else {
                    best_lower_sum - initial_lower_sum
                };

                if pilot_improvement < config.pilot_improvement_threshold {
                    info!(
                        "DAG α-CROWN: Pilot iteration improvement ({:.3e}) < threshold ({:.3e}). \
                         α-CROWN optimization is not helping, skipping remaining iterations.",
                        pilot_improvement, config.pilot_improvement_threshold
                    );
                    // Return best bounds found so far (CROWN or pilot iteration bounds)
                    let mut pilot_lower = best_lower.clone();
                    let mut pilot_upper = best_upper.clone();
                    let widened = clamp_inverted_best_bounds(
                        &mut pilot_lower,
                        &mut pilot_upper,
                        "dag-alpha-crown-pilot-exit",
                    );
                    if widened > 0 {
                        // Fall back to CROWN bounds for inverted elements (#3754).
                        for (best, &crown) in
                            pilot_lower.iter_mut().zip(crown_bounds.lower().iter())
                        {
                            if !best.is_finite() {
                                *best = crown;
                            }
                        }
                        for (best, &crown) in
                            pilot_upper.iter_mut().zip(crown_bounds.upper().iter())
                        {
                            if !best.is_finite() {
                                *best = crown;
                            }
                        }
                    }
                    let bounds = BoundedTensor::new(pilot_lower, pilot_upper).map_err(|e| {
                        NyError::InternalError(format!(
                            "DAG α-CROWN pilot bounds invalid after CROWN fallback: {e}"
                        ))
                    })?;
                    // #phase-telemetry: this pilot skip returns without
                    // reaching the shared post-loop exit marker below.
                    if crate::phase_telemetry::phase_telemetry_enabled() {
                        crate::phase_telemetry::phase_marker(&format!(
                            "dag-alpha-warmup loop-exit iters={phase_iters_started} (pilot-skip)"
                        ));
                    }
                    // #root-alpha-margin: the pilot return bypasses the shared
                    // post-loop restoration below. Preserve the same whole
                    // GraphAlphaState checkpoint contract on this exit instead
                    // of accidentally returning the current (possibly worse)
                    // iterate. With no scored checkpoint this is exactly the
                    // legacy current-state snapshot.
                    let alpha_state = best_margin_alpha
                        .as_ref()
                        .map_or_else(|| runtime.snapshot_graph(), Clone::clone);
                    return Ok(DagAlphaCollectionArtifact {
                        output_bounds: bounds,
                        alpha_state,
                        reference_bounds: reference_bounds.into_current(),
                        reference_bounds_source: node_bounds_source,
                        completed_iterations: phase_iters_completed,
                        optimizer_updates_completed: best_margin_alpha
                            .as_ref()
                            .map_or(phase_optimizer_updates_completed, |_| best_margin_iter),
                    });
                } else {
                    debug!(
                        "DAG α-CROWN: Pilot iteration improvement ({:.3e}) >= threshold ({:.3e}). \
                         Continuing optimization.",
                        pilot_improvement, config.pilot_improvement_threshold
                    );
                }
            }

            // All terminal bound validity, best-bound, early-stop, and pilot
            // bookkeeping is complete. Preserve the exact alpha/reference/
            // optimizer state that produced it; nothing below can feed another
            // evaluated bound.
            if !need_grad {
                debug!(
                    method = ?config.gradient_method,
                    iter,
                    skipped_gradient_dispatches = 1usize,
                    skipped_state_updates = 1usize,
                    "DAG α-CROWN: NY_ALPHA_FINAL_BOUND_ONLY terminal pass"
                );
                break;
            }

            // A pure linear gamma-only episode has no alpha-bearing state.
            // The gamma transaction above is its complete optimizer update;
            // numerical/analytic alpha gradient dispatch would only pay for an
            // additional full backward and then mutate nothing.
            if no_alpha_optimizer {
                if !gamma_only_optimizer
                    || !config.invprop.optimize_gammas
                    || invprop_gap_score.is_none()
                    || iter >= invprop_ascent_max_iters
                {
                    break;
                }
                lr *= config.lr_decay;
                prev_best_lower_sum = best_lower_sum;
                prev_best_invprop_gap = best_invprop_gap;
                prev_best_margin_score = best_margin_score;
                previous_margin_joint_dispatched = false;
                continue;
            }

            // #joint-interm-alpha: decide whether to REBUILD the relaxation at the
            // current alpha this iteration.
            //
            // Legacy (`joint_interm_alpha_every == 0`) keeps the historical
            // `improved_output` gate, byte-identical. That gate is measured DEAD
            // on cifar100 for two independent reasons, which is why the joint
            // mode does not reuse it:
            //
            //   1. `improved_output` is `lower_sum > best_lower_sum` where
            //      `lower_sum` is the plain sum over the 100 RAW LOGITS, and
            //      `best_lower_sum` is seeded from the PRE-loop CROWN bound. The
            //      measured trace goes the wrong way (-152.2 at 1 iteration vs
            //      -555.8 at 10), so the condition is false from iteration 0 on.
            //   2. Even were it well-signed, the shipped alpha gradient is
            //      sign-definite <= 0 (machine-checked in AlphaGradientDefect.lean),
            //      so the ascent has nothing to improve WITH.
            //
            // Gating the rebuild on "did alpha improve the bound" therefore makes
            // the relaxation refresh conditional on the very thing a frozen
            // relaxation prevents. Joint mode cuts that circular dependency: it
            // rebuilds on a fixed cadence and lets the tighter relaxation create
            // the improvement, which is what block-coordinate ascent requires.
            let joint_every = config.joint_interm_alpha_every;
            let joint_due = joint_every > 0 && iter.is_multiple_of(joint_every);
            let rebuild_due = if joint_every > 0 {
                joint_due
            } else {
                improved_output
            };
            let refresh_candidate = if iter >= 1
                && rebuild_due
                && !reference_bounds.targets().is_empty()
            {
                // Carry forward tighter activation-input bounds between
                // iterations, matching alpha-beta-CROWN's
                // `best_intermediate_bounds` / `reference_bounds`
                // tightening for optimizable activations.
                // Source: auto_LiRPA `optimized_bounds.py:338-367,500-615`.
                //
                // #w4-gpu-dag-backward: bound the refresh to a SHARE of the
                // remaining budget instead of the full global deadline.
                // Measured (cifar100 resnet-medium, release): one unbounded
                // refresh ran ~120s of per-target spec-batched CROWN requests
                // — past the whole 95s timeout — while a GPU warmup iteration
                // costs ~1.5s, so a single refresh starved every remaining
                // iteration AND BaB. On expiry the refresh falls back to the
                // previous (sound) reference bounds for outstanding targets,
                // so capping only trades tightness for schedule — never
                // soundness.
                // #wall-airtime: all improving iterations draw from ONE
                // cumulative pool. The typed AlphaCrownConfig supplies its
                // fraction and optional absolute ceiling; the historical
                // NY_ALPHA_REFRESH_FRACTION remains a measurement override
                // for the fraction only.
                let refresh_start = std::time::Instant::now();
                let global_remaining = config.deadline.map(|global_deadline| {
                    global_deadline
                        .checked_duration_since(refresh_start)
                        .unwrap_or_default()
                });
                let allowance = cumulative_alpha_refresh_allowance(
                    &mut alpha_refresh_budget_remaining,
                    global_remaining,
                    alpha_refresh_fraction,
                    alpha_refresh_max_duration,
                );
                let refresh_deadline = allowance.map(|allowance| {
                    refresh_start
                        .checked_add(allowance)
                        .unwrap_or(refresh_start)
                });
                let has_refresh_budget = allowance.is_none_or(|allowance| !allowance.is_zero());
                if has_refresh_budget {
                    // #joint-interm-alpha: on a cadence the expensive face is not
                    // affordable. `targets_only=false` walks the WHOLE exec order
                    // rebuilding a crown cache so later nodes relax off earlier
                    // ones — measured at ~120s on cifar100 resnet-medium against a
                    // 40s root-alpha cap, i.e. one refresh consumes the entire
                    // ascent. `targets_only=true` walks only the targets and uses
                    // the reference map as the relaxation source elsewhere: looser
                    // (no cascade between targets within one pass) but O(#targets)
                    // instead of O(L^2), and the cadence recovers the cascade
                    // across iterations instead of within one.
                    //
                    // Both faces commit through the same shrink-only
                    // `merge_tighter_bounds`, so the cheap face can only be less
                    // tight, never unsound.
                    let targets_only = joint_every > 0;
                    let candidate = match self.collect_selected_crown_bounds_with_alpha_mode(
                        input,
                        reference_bounds.targets(),
                        node_bounds,
                        runtime.graph(),
                        engine,
                        refresh_deadline,
                        targets_only,
                    ) {
                        Ok(candidate) => candidate,
                        Err(error)
                            if retain_completed_deadline_error(
                                retain_completed_on_deadline,
                                phase_iters_completed,
                                &error,
                            ) =>
                        {
                            info!(
                                    completed_iterations = phase_iters_completed,
                                    "DAG α-CROWN: retaining completed artifact after reference refresh reached a deadline"
                                );
                            break 'alpha_iterations;
                        }
                        Err(error) => return Err(error),
                    };
                    debit_alpha_refresh_budget(
                        &mut alpha_refresh_budget_remaining,
                        refresh_start.elapsed(),
                    );
                    Some(candidate)
                } else {
                    debug!(
                        iter,
                        "DAG α-CROWN: cumulative reference-refresh budget exhausted"
                    );
                    None
                }
            } else {
                None
            };

            // #root-alpha-gpu (B): a reference-bound refresh ran this
            // iteration — invalidate the loop-top fold's gradient reuse so the
            // gradient site re-folds fresh (a hygiene choice, not soundness:
            // gradients only steer α and never decide a verdict).
            if refresh_candidate.is_some() {
                if let Some(cache) = warmup_iter_cache.as_mut() {
                    cache.refresh_fired = true;
                }
            }

            // Compute gradients using configured method (SPSA, FD, Analytic, AnalyticChain).
            let eps = 1e-3;
            let margin_request = if margin_gradient_eligible {
                margin_gradient_objective.as_ref().map_or(
                    gradients::MarginGradientRequest::NoBinding,
                    |binding| {
                        gradients::MarginGradientRequest::Binding(
                            binding.lower_objective.as_slice(),
                        )
                    },
                )
            } else {
                gradients::MarginGradientRequest::Disabled
            };
            // #replay-row-index PROBE (dark, `NY_ALPHA_REPLAY_ROWIDX=1`,
            // default OFF ⇒ byte-identical). MEASURED on cifar100 idx_7704:
            // `#binding-row-replay` refuses EVERY iteration with
            // `objective_not_single_positive_row`, because the binding
            // objective is a MARGIN row (y_i − y_j ⇒ two nonzeros in logit
            // space) while the admission test demands a scaled `e_r`. The
            // captured seed rows are the spec rows, so the row index the
            // ascent already knows is the seed row the replay needs. Publish
            // it here so the replay can index directly instead of decoding
            // logit-space coefficients.
            gradients::set_binding_seed_row(
                margin_gradient_objective.as_ref().map(|b| b.row_index),
            );
            let mut grad_result = match self.compute_dag_gradients(
                &ctx,
                node_bounds,
                &mut runtime,
                &mut bilinear_alphas,
                &mut mul_binary_alphas,
                &gradients,
                &gradients_upper,
                eps,
                iter,
                warmup_iter_cache.as_ref(),
                margin_request,
            ) {
                Ok(result) => result,
                Err(error)
                    if retain_completed_deadline_error(
                        retain_completed_on_deadline,
                        phase_iters_completed,
                        &error,
                    ) =>
                {
                    info!(
                        completed_iterations = phase_iters_completed,
                        "DAG α-CROWN: retaining completed artifact after gradient dispatch reached the phase deadline"
                    );
                    break 'alpha_iterations;
                }
                Err(error) => return Err(error),
            };
            if let Err(error) = self.compute_spsa_supplements(
                input,
                node_bounds,
                &exec_order,
                output_dim,
                input_dim,
                config,
                engine,
                &mut runtime,
                &gradients,
                &bilinear_alphas,
                &mut mul_binary_alphas,
                &mut grad_result.mul_binary_grads,
                &mut grad_result.s_shaped_grads,
                &mut grad_result.sqrt_grads,
                &mut grad_result.reciprocal_grads,
                has_bilinear,
                has_mul_binary,
                has_s_shaped,
                has_sqrt,
                has_reciprocal,
                eps,
                iter,
            ) {
                if retain_completed_deadline_error(
                    retain_completed_on_deadline,
                    phase_iters_completed,
                    &error,
                ) {
                    info!(
                        completed_iterations = phase_iters_completed,
                        "DAG α-CROWN: retaining completed artifact after gradient supplement reached the phase deadline"
                    );
                    break 'alpha_iterations;
                }
                return Err(error);
            }

            // Destructure to separate immutable and mutable borrows (#2297).
            // `numerical_gradients_upper` is `None` for non-Analytic methods,
            // avoiding a full Vec<Array1<f32>> clone per iteration.
            let gradients::GradientDispatchResult {
                numerical_gradients: ref lower_grads,
                numerical_gradients_upper: ref upper_grads_opt,
                ref s_shaped_grads,
                ref sqrt_grads,
                ref reciprocal_grads,
                bilinear_grads: ref mut bl_grads,
                mul_binary_grads: ref mut mb_grads,
                margin_dispatch,
            } = grad_result;
            if margin_gradient_eligible {
                match margin_dispatch {
                    // The resident (verdict-authority backend) line stays
                    // byte-identical to the pre-proposal telemetry; the
                    // proposal channel reports its own source variant beside
                    // supplied_local/resident_local_rule
                    // (#alpha-steering-proposal).
                    gradients::MarginGradientDispatch::JointDispatched {
                        source: gradients::MarginGradientJointSource::Resident,
                    } => info!(
                        "DAG α-CROWN #root-alpha-margin-gradient iter {iter}: \
                         dispatch=joint_resident"
                    ),
                    // #binding-row-replay: sibling of the proposal line —
                    // the deterministic CPU replay consumed as the margin
                    // gradient on a non-resident host.
                    gradients::MarginGradientDispatch::JointDispatched {
                        source: gradients::MarginGradientJointSource::CpuReplay,
                    } => info!(
                        "DAG α-CROWN #root-alpha-margin-gradient iter {iter}: \
                         dispatch=binding_row_replay source=cpu_replay"
                    ),
                    gradients::MarginGradientDispatch::JointDispatched { source } => info!(
                        "DAG α-CROWN #root-alpha-margin-gradient iter {iter}: \
                         dispatch=joint_proposal source={}",
                        source.as_str()
                    ),
                    gradients::MarginGradientDispatch::LocalFallback { reason, source } => info!(
                        "DAG α-CROWN #root-alpha-margin-gradient iter {iter}: \
                         dispatch=local_fallback reason={} source={}",
                        reason.as_str(),
                        source.as_str()
                    ),
                    gradients::MarginGradientDispatch::NotRequested => info!(
                        "DAG α-CROWN #root-alpha-margin-gradient iter {iter}: \
                         dispatch=internal_not_requested"
                    ),
                }
            }
            let upper_grads: &[Array1<f32>] = upper_grads_opt.as_deref().unwrap_or(lower_grads);
            // #binding-row-replay trust region (consult #6 v1 acceptance
            // policy, the cheap half only): replay-sourced updates are
            // projected to max|Δα| ≤ 0.05 AFTER the optimizer runs — a
            // projection around `update_all_alphas`, never a change to the
            // updater itself. Other sources (resident/proposal/local) are
            // byte-identical to today. The transactional-rollback half is a
            // later slice.
            let replay_trust_region_snapshot = matches!(
                margin_dispatch,
                gradients::MarginGradientDispatch::JointDispatched {
                    source: gradients::MarginGradientJointSource::CpuReplay,
                }
            )
            .then(|| snapshot_relu_alpha_pairs(runtime.graph()));
            alpha_update::update_all_alphas(
                &mut runtime,
                config,
                lower_grads,
                upper_grads,
                s_shaped_grads,
                sqrt_grads,
                reciprocal_grads,
                &mut bilinear_alphas,
                bl_grads,
                &mut bilinear_adam_m,
                &mut bilinear_adam_v,
                &mut mul_binary_alphas,
                mb_grads,
                &mut mul_binary_adam_m,
                &mut mul_binary_adam_v,
                has_bilinear,
                has_mul_binary,
                invprop_enabled,
                lr,
                iter,
                &mut total_gradient_skips,
            )?;
            phase_optimizer_updates_completed = iter + 1;
            if let Some(snapshot) = replay_trust_region_snapshot.as_ref() {
                let clamped = clamp_relu_alpha_trust_region(runtime.graph_mut(), snapshot);
                if clamped > 0 {
                    debug!(
                        "DAG α-CROWN #binding-row-replay iter {iter}: trust region \
                         clamped {clamped} alpha components to |Δα| ≤ {REPLAY_ALPHA_TRUST_REGION}"
                    );
                }
            }

            // #spec-axis-alpha slice 2d: when the margin lane bound a row that
            // owns a δ slot, this iteration's per-ReLU gradients ARE that
            // row's ∂bound/∂α — feed them to the slot's δ on top of the shared
            // update. Rows without slots change nothing; no margin binding ⇒
            // δ rests. Sound by construction: δ only re-parameterizes which
            // valid α ∈ [0,1] the next certified fold evaluates.
            if config.alpha_spec_slots > 0 {
                if let Some(binding) = margin_gradient_objective.as_ref() {
                    // Slice 2e: the margin lane's binding criterion (its own
                    // slack rule) diverges from iter-0 worst-margin
                    // assignment — MEASURED on prop_idx_7500: slots
                    // [71,68,75,20,58,69,90,95], binding rows {48,21},
                    // intersection empty, δ never updated. A binding row
                    // without a slot CLAIMS one by round-robin reassignment;
                    // δ and moments reset to the parity anchor, so the
                    // incoming row starts clean and the outgoing row's
                    // corrections never leak.
                    if runtime
                        .graph()
                        .slot_for_spec_row(binding.row_index)
                        .is_none()
                    {
                        let slots = runtime.graph().spec_slot_rows.len();
                        if slots > 0 {
                            let victim = iter % slots;
                            if runtime
                                .graph_mut()
                                .reassign_spec_slot(victim, binding.row_index)
                            {
                                info!(
                                    "DAG α-CROWN #spec-axis-alpha iter {iter}: slot {victim} \
                                     reassigned to binding row {}",
                                    binding.row_index
                                );
                            }
                        }
                    }
                    if let Some(slot) = runtime.graph().slot_for_spec_row(binding.row_index) {
                        let slots = runtime.graph().spec_slot_rows.len();
                        let adam = crate::bounds::AdamParams {
                            learning_rate: lr,
                            beta1: 0.9,
                            beta2: 0.999,
                            epsilon: 1e-8,
                            t: iter + 1,
                        };
                        let named: Vec<(String, usize)> = runtime
                            .relu_name_to_idx()
                            .iter()
                            .map(|(name, &idx)| (name.clone(), idx))
                            .collect();
                        for (name, idx) in named {
                            let Some(grad) = lower_grads.get(idx) else {
                                continue;
                            };
                            let width = match runtime.graph().alpha(&name) {
                                Some(alpha) if alpha.len() == grad.len() => alpha.len(),
                                _ => continue, // channel/width mismatch ⇒ skip node
                            };
                            let mut per_slot = ndarray::Array2::<f32>::zeros((slots, width));
                            per_slot.row_mut(slot).assign(grad);
                            runtime.graph_mut().update_spec_deltas_adam(
                                &name,
                                &per_slot,
                                iter + 1,
                                &adam,
                            );
                        }
                    }
                }
            }

            if let Some(candidate) = refresh_candidate {
                let tightened_targets = reference_bounds.merge_candidate(&candidate)?;
                reference_bounds.promote_best_to_current()?;
                debug!(
                    "DAG α-CROWN iter {}: refreshed {} activation-input reference targets",
                    iter, tightened_targets
                );
                // #joint-interm-alpha RULE 7: the ladder this belongs to has
                // repeatedly produced nulls that were really inert arms — a lane
                // that reported `gate ON` and then `0/6 tightened`, a lever whose
                // telemetry was empty in both arms, and a set-mate fix whose probe
                // never fired. So the joint arm states, unconditionally and on
                // stderr, that it RAN and what it achieved. A null is only
                // believable against a non-empty line here, and `tightened=0` is a
                // materially different claim from silence.
                if joint_every > 0 {
                    eprintln!(
                        "[joint-interm-alpha] iter={iter} cadence={joint_every} \
                         targets={} tightened={tightened_targets} targets_only=true",
                        reference_bounds.targets().len(),
                    );
                }
            }

            // Learning rate decay
            lr *= config.lr_decay;

            if iter % 5 == 0 {
                diagnostics::log_iteration_telemetry(
                    &runtime,
                    iter,
                    best_lower_sum,
                    prev_best_lower_sum,
                    lower_sum,
                    lr,
                );
            }

            // #invprop-alpha-budget: info-level progress heartbeat. The alpha
            // phase used to emit NOTHING at default log level between the
            // "Starting optimization" line and its exit — on slow models it
            // was indistinguishable from a hang. Every 10 iterations, report
            // position and remaining budget.
            if iter % 10 == 0 {
                match config
                    .deadline
                    .map(|d| d.saturating_duration_since(std::time::Instant::now()))
                {
                    Some(remaining) => info!(
                        "DAG α-CROWN: iter {}/{}: lower_sum={:.6}, best_lower_sum={:.6}, \
                         budget remaining {:.1}s",
                        iter,
                        config.iterations,
                        lower_sum,
                        best_lower_sum,
                        remaining.as_secs_f32()
                    ),
                    None => info!(
                        "DAG α-CROWN: iter {}/{}: lower_sum={:.6}, best_lower_sum={:.6}",
                        iter, config.iterations, lower_sum, best_lower_sum
                    ),
                }
            }

            prev_best_lower_sum = best_lower_sum;
            prev_best_invprop_gap = best_invprop_gap;
            prev_best_margin_score = best_margin_score;
            previous_margin_joint_dispatched = margin_dispatch.joint_dispatched();
        }

        // #phase-telemetry: shared loop exit (normal completion and every
        // `break` — deadline, converged, NaN, spec-early-exit, infeasible).
        // `?` error exits skip it, but those abort the pipeline anyway.
        if crate::phase_telemetry::phase_telemetry_enabled() {
            crate::phase_telemetry::phase_marker(&format!(
                "dag-alpha-warmup loop-exit iters={phase_iters_started}"
            ));
        }

        diagnostics::log_gradient_skip_summary(
            total_gradient_skips,
            config.iterations,
            runtime.relu_nodes().len(),
        );

        // Select the alpha checkpoint that will be returned before any final
        // zero-gamma reconstruction. The accumulated best box may combine
        // several valid global iterates, but the recovery fold must evaluate
        // the same selected checkpoint handed downstream as its warm start.
        if infeasible_bounds.is_none() {
            if let Some(best_alpha) = best_margin_alpha.take() {
                info!(
                    "DAG α-CROWN #root-alpha-margin: restoring α from iteration {best_margin_iter} \
                     (hinge {:.6}) instead of the last iterate",
                    best_margin_score.unwrap_or(f32::NAN),
                );
                runtime.restore_graph(&best_alpha);
                phase_optimizer_updates_completed = best_margin_iter;
            }
        }

        // #gap-attribution (build-plan step 3): attribute the settled shared
        // root checkpoint after any best-margin restore. This is intentionally
        // a STATIC advisory prior, not a claim about the later per-domain BaB
        // alpha (root assembly may replace or discard that warm start). The
        // theorem is exact for this directly seeded fold; the optional scores
        // only steer branching. It touches no bound/verdict and contains every
        // refusal or panic.
        if crate::network::graph_alpha::gap_attribution::root_gap_probe_enabled() {
            crate::network::graph_alpha::gap_attribution::run_root_gap_probe(
                self,
                input,
                // `node_bounds` moved into the reference-bounds holder before
                // the loop; `current()` is what every in-loop fold reads.
                reference_bounds.current(),
                &exec_order,
                output_dim,
                input_dim,
                runtime.relu_name_to_idx(),
                runtime.graph(),
                engine,
                crate::network::graph_alpha::gap_attribution::attribution_run_deadline(),
                &best_lower,
                &best_upper,
                config.spec_ascent.as_ref(),
            );
        }

        // Conditioned gamma iterates cannot enter the global output box, but
        // their alpha/extended-alpha updates remain globally valid relaxation
        // parameters. If gamma did not prove emptiness, reset only the output
        // seed to exact zero for one deadline-gated bound-only fold, merge that
        // reconstructed global bound, then restore the optimized seed on every
        // outcome. Optional probe-style failures retain the initial global
        // CROWN bound already present in `best_*`.
        if infeasible_bounds.is_none()
            && invprop_gamma_optimization_active
            && !no_alpha_optimizer
            && !config.past_deadline()
        {
            let seed_key = output_node_name.unwrap_or_default();
            let base_params = runtime
                .invprop()
                .and_then(|state| state.layer_gammas(seed_key))
                .filter(|gammas| gammas.active)
                .map(|gammas| gammas.gammas.clone());
            if let Some(base_params) = base_params.filter(|params| {
                params
                    .iter()
                    .any(|value| value.to_bits() != 0.0_f32.to_bits())
            }) {
                let zero_params = Array3::zeros(base_params.dim());
                install_invprop_seed_params(&mut runtime, seed_key, &zero_params)?;
                let globally_unconditioned = runtime.invprop().is_some_and(|state| {
                    state
                        .all_ny_params()
                        .iter()
                        .all(|value| value.to_bits() == 0.0_f32.to_bits())
                });
                let (bilinear_ref, mul_binary_ref) =
                    gradients::alpha_refs(&ctx, &bilinear_alphas, &mul_binary_alphas);
                let final_result = globally_unconditioned.then(|| {
                    self.dag_alpha_bound_pass_with_engine_and_infeasibility(
                        input,
                        reference_bounds.current(),
                        &exec_order,
                        output_dim,
                        input_dim,
                        runtime.relu_name_to_idx(),
                        runtime.graph(),
                        // Exact-zero gamma is the identity. Omitting the state
                        // preserves the ordinary concretization/GPU suffix path
                        // and cannot change the reconstructed global bound.
                        None,
                        engine,
                        bilinear_ref,
                        mul_binary_ref,
                        config.deadline,
                    )
                });
                install_invprop_seed_params(&mut runtime, seed_key, &base_params)?;
                match final_result {
                    // This transaction proved every gamma exactly zero before
                    // dispatch, so GPU-suffix and conservative fallback results
                    // are also valid global bounds. `completed` is proof-critical
                    // only for conditioned folds and must not discard them here.
                    Some(Ok((global_bounds, _proof, _gap, _row_gaps, _completed)))
                        if !global_bounds.lower().iter().any(|value| value.is_nan())
                            && !global_bounds.upper().iter().any(|value| value.is_nan()) =>
                    {
                        #[cfg(test)]
                        if global_bounds
                            .lower()
                            .iter()
                            .zip(best_lower.iter())
                            .any(|(&candidate, &best)| candidate > best)
                            || global_bounds
                                .upper()
                                .iter()
                                .zip(best_upper.iter())
                                .any(|(&candidate, &best)| candidate < best)
                        {
                            INVPROP_ZERO_GAMMA_RECOVERY_IMPROVEMENTS
                                .with(|slot| slot.set(slot.get() + 1));
                        }
                        update_elementwise_best_bounds(
                            &mut best_lower,
                            &mut best_upper,
                            &global_bounds,
                            config.iterations,
                        )?;
                    }
                    // A NaN-carrying reconstruction is discarded, exactly as
                    // the guarded arm's condition did before it was collapsed.
                    Some(Ok(_)) => {}
                    None => {}
                    Some(Err(error)) if invprop_gamma_probe_can_noop(&error) => {}
                    Some(Err(error))
                        if retain_completed_deadline_error(
                            retain_completed_on_deadline,
                            phase_iters_completed,
                            &error,
                        ) =>
                    {
                        info!(
                            completed_iterations = phase_iters_completed,
                            "DAG α-CROWN: retaining completed artifact after final global reconstruction reached the phase deadline"
                        );
                    }
                    Some(Err(error)) => {
                        return Err(error);
                    }
                }
            }
        }

        if let Some(bounds) = infeasible_bounds {
            return Ok(DagAlphaCollectionArtifact {
                output_bounds: bounds,
                alpha_state: runtime.into_graph_alpha_state(),
                reference_bounds: reference_bounds.into_current(),
                reference_bounds_source: node_bounds_source,
                completed_iterations: phase_iters_completed,
                optimizer_updates_completed: phase_optimizer_updates_completed,
            });
        }

        // Return element-wise best bounds found across all iterations.
        // Fall back to CROWN only when NaN is present (computation error).
        // Infinite bounds are sound overapproximations from inversion widening
        // (clamp_inverted_best_bounds sets inverted elements to [-inf, +inf]).
        // The previous is_finite() check incorrectly discarded ALL optimization
        // progress when any single element was infinite (#2854).
        let alpha_state = runtime.into_graph_alpha_state();
        let reference_bounds = reference_bounds.into_current();
        let has_nan =
            best_lower.iter().any(|v| v.is_nan()) || best_upper.iter().any(|v| v.is_nan());

        if !has_nan {
            // Clamp any inverted intervals from cross-iteration elementwise merge.
            let widened =
                clamp_inverted_best_bounds(&mut best_lower, &mut best_upper, "dag-alpha-crown");

            if widened > 0 {
                // Fall back to CROWN bounds for inverted elements (#3754).
                // Cross-iteration elementwise merge can produce inversions on DAG
                // topologies (e.g. diamond: branches with zero weights cause SPSA
                // noise to oscillate alpha slopes). The initial CROWN bounds are
                // sound and finite for finite-weight networks, so restoring them
                // for widened elements preserves soundness while avoiding -inf.
                for (best, &crown) in best_lower.iter_mut().zip(crown_bounds.lower().iter()) {
                    if !best.is_finite() {
                        *best = crown;
                    }
                }
                for (best, &crown) in best_upper.iter_mut().zip(crown_bounds.upper().iter()) {
                    if !best.is_finite() {
                        *best = crown;
                    }
                }
            }

            let bounds = BoundedTensor::new(best_lower, best_upper).map_err(|e| {
                NyError::InternalError(format!(
                    "DAG α-CROWN best bounds invalid after CROWN fallback: {e}"
                ))
            })?;
            Ok(DagAlphaCollectionArtifact {
                output_bounds: bounds,
                alpha_state,
                reference_bounds,
                reference_bounds_source: node_bounds_source,
                completed_iterations: phase_iters_completed,
                optimizer_updates_completed: phase_optimizer_updates_completed,
            })
        } else {
            // Fall back to CROWN if NaN detected (actual computation error)
            warn!("DAG α-CROWN: NaN in best bounds, falling back to plain CROWN");
            let bounds = match self
                .propagate_crown_with_engine_and_deadline(input, engine, config.deadline)
                .map(|r| r.bounds)
            {
                Ok(bounds) => bounds,
                Err(error)
                    if retain_completed_deadline_error(
                        retain_completed_on_deadline,
                        phase_iters_completed,
                        &error,
                    ) =>
                {
                    warn!(
                        completed_iterations = phase_iters_completed,
                        "DAG α-CROWN: phase deadline prevented NaN fallback recollection; retaining the pre-loop certified CROWN output"
                    );
                    crown_bounds
                }
                Err(error) => return Err(error),
            };
            Ok(DagAlphaCollectionArtifact {
                output_bounds: bounds,
                alpha_state,
                reference_bounds,
                reference_bounds_source: node_bounds_source,
                completed_iterations: phase_iters_completed,
                optimizer_updates_completed: phase_optimizer_updates_completed,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        alpha_zero_yield_frac_from, binding_margin_lower_objective, can_reuse_initial_node_bounds,
        cumulative_alpha_refresh_allowance, debit_alpha_refresh_budget,
        hinge_margin_lower_objective, margin_gradient_eligibility, parse_alpha_divergence_bail,
        parse_alpha_refresh_fraction, parse_root_alpha_margin, parse_root_alpha_margin_gradient,
        parse_root_alpha_margin_hinge, retain_completed_deadline_error, retain_warmup_iter_cache,
        root_alpha_margin_enabled_from, root_alpha_margin_gradient_enabled_if,
        root_alpha_margin_hinge_enabled, root_alpha_margin_state_from, AlphaReferenceBoundsSource,
        DagAlphaLoopResultUse, MarginGradientEligibility, ALPHA_DIVERGENCE_FACTOR,
    };
    use crate::bounds::{AlphaCrownConfig, AlphaSpecAscent, AlphaSpecEarlyExit};
    use std::cell::Cell;
    use std::time::Duration;

    #[test]
    fn completed_deadline_retention_requires_policy_fold_and_deadline_error() {
        let deadline = ny_core::NyError::DeadlineExceeded("phase cap".into());
        assert!(retain_completed_deadline_error(true, 1, &deadline));
        assert!(!retain_completed_deadline_error(false, 1, &deadline));
        assert!(!retain_completed_deadline_error(true, 0, &deadline));
        assert!(!retain_completed_deadline_error(
            true,
            1,
            &ny_core::NyError::InvalidSpec("not a deadline".into()),
        ));
    }

    /// #binding-row-replay trust region (consult #6 v1, cheap half): a
    /// synthetic huge gradient through the REAL updater (Adam, the default
    /// lr 0.1 step overshoots the region) is projected back to
    /// max|Δα| EXACTLY 0.05 from the pre-update iterate — on the lower AND
    /// upper α paths — while in-region components are untouched bitwise.
    #[test]
    fn replay_trust_region_clamps_huge_gradient_to_exactly_five_hundredths() {
        use ndarray::{Array1, ArrayD, IxDyn};
        use ny_tensor::BoundedTensor;

        let width = 6usize;
        let pre = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[width]), -1.0f32),
            ArrayD::from_elem(IxDyn(&[width]), 1.0f32),
        )
        .expect("unstable pre-activation box");
        let mut state = super::GraphAlphaState::new();
        state
            .add_relu_node("relu0", &pre, false)
            .expect("relu state");
        if let Some((lower, upper)) = state.relu_alpha_pair_mut("relu0") {
            lower.fill(0.5);
            upper.fill(0.5);
        }
        let snapshot = super::snapshot_relu_alpha_pairs(&state);

        // Huge-magnitude gradient → the Adam step saturates at ~lr = 0.1.
        let huge = Array1::from_elem(width, 1e9f32);
        let adam = crate::bounds::AdamParams {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 1,
        };
        state.update_adam("relu0", &huge, &adam);
        state.update_adam_upper("relu0", &huge.mapv(|value| -value), &adam);

        // Sanity: the raw update must actually overshoot the trust region.
        {
            let (lower, upper) = state.relu_alpha_pair("relu0").expect("alpha pair");
            assert!(
                lower.iter().any(|&value| (value - 0.5).abs() > 0.05)
                    || upper.iter().any(|&value| (value - 0.5).abs() > 0.05),
                "fixture must overshoot: got lower {lower:?} upper {upper:?}"
            );
        }

        let clamped = super::clamp_relu_alpha_trust_region(&mut state, &snapshot);
        assert!(clamped > 0, "the projection must have clamped something");
        // The projection endpoints in the projection's own f32 arithmetic:
        // `anchor ± 0.05` — "exactly 0.05" means landing bitwise on these
        // (the recomputed f32 difference re-rounds by up to one ulp, e.g.
        // fl(0.45) − 0.5 = −0.050000012).
        let expected_lo = 0.5f32 + super::REPLAY_ALPHA_TRUST_REGION;
        let expected_hi = 0.5f32 - super::REPLAY_ALPHA_TRUST_REGION;
        let region = (expected_lo - 0.5f32)
            .abs()
            .max((expected_hi - 0.5f32).abs());
        let (lower, upper) = state.relu_alpha_pair("relu0").expect("alpha pair");
        for value in lower.iter().chain(upper.iter()) {
            let delta = value - 0.5f32;
            assert!(
                delta.abs() <= region,
                "post-projection |Δα| must be ≤ the 0.05 projection endpoint, got {delta}"
            );
        }
        // The saturated components sit EXACTLY at anchor ± 0.05 (bitwise).
        for value in lower.iter() {
            assert!(
                value.to_bits() == expected_lo.to_bits()
                    || value.to_bits() == expected_hi.to_bits(),
                "saturated lower α must land exactly at 0.5 ± 0.05, got {value}"
            );
        }
        for value in upper.iter() {
            assert!(
                value.to_bits() == expected_lo.to_bits()
                    || value.to_bits() == expected_hi.to_bits(),
                "saturated upper α must land exactly at 0.5 ± 0.05, got {value}"
            );
        }

        // In-region updates pass through bitwise untouched.
        let mut untouched = super::GraphAlphaState::new();
        untouched
            .add_relu_node("relu0", &pre, false)
            .expect("relu state");
        if let Some((lower, upper)) = untouched.relu_alpha_pair_mut("relu0") {
            lower.fill(0.5);
            upper.fill(0.5);
        }
        let snapshot = super::snapshot_relu_alpha_pairs(&untouched);
        if let Some((lower, upper)) = untouched.relu_alpha_pair_mut("relu0") {
            lower.fill(0.52);
            upper.fill(0.47);
        }
        assert_eq!(
            super::clamp_relu_alpha_trust_region(&mut untouched, &snapshot),
            0,
            "in-region moves must not be clamped"
        );
        let (lower, upper) = untouched.relu_alpha_pair("relu0").expect("alpha pair");
        assert!(lower
            .iter()
            .all(|value| value.to_bits() == 0.52f32.to_bits()));
        assert!(upper
            .iter()
            .all(|value| value.to_bits() == 0.47f32.to_bits()));
    }

    /// ADVERSARIAL clamp edge cases (#binding-row-replay trust region):
    /// - a `-0.0` anchor/value never trips the clamp for in-region moves and
    ///   is passed through bit-identically (`-0.0 - 0.0 == -0.0`, not a move);
    /// - a NaN post-update component passes through UNCLAMPED and unchanged
    ///   (both `delta > r` and `delta < -r` are false for NaN) — the
    ///   projection fails open exactly like its width-mismatch arm, never
    ///   manufactures a value, never panics, and reports 0 clamps for it;
    /// - an overshoot from a `-0.0` anchor still lands exactly at
    ///   `anchor + 0.05`.
    #[test]
    fn replay_trust_region_clamp_edge_cases_nan_and_negative_zero() {
        use ndarray::{ArrayD, IxDyn};
        use ny_tensor::BoundedTensor;

        let width = 3usize;
        let pre = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[width]), -1.0f32),
            ArrayD::from_elem(IxDyn(&[width]), 1.0f32),
        )
        .expect("unstable pre-activation box");
        let mut state = super::GraphAlphaState::new();
        state
            .add_relu_node("relu0", &pre, false)
            .expect("relu state");
        if let Some((lower, upper)) = state.relu_alpha_pair_mut("relu0") {
            lower[0] = -0.0;
            lower[1] = 0.5;
            lower[2] = -0.0;
            upper.fill(0.5);
        }
        let snapshot = super::snapshot_relu_alpha_pairs(&state);
        if let Some((lower, upper)) = state.relu_alpha_pair_mut("relu0") {
            lower[0] = -0.0; // unchanged -0.0: not a move
            lower[1] = f32::NAN; // poisoned post-update component
            lower[2] = 0.2; // overshoot from a -0.0 anchor
            upper.fill(0.5); // untouched path
        }
        let clamped = super::clamp_relu_alpha_trust_region(&mut state, &snapshot);
        assert_eq!(clamped, 1, "only the overshoot component may clamp");
        let (lower, upper) = state.relu_alpha_pair("relu0").expect("alpha pair");
        assert_eq!(
            lower[0].to_bits(),
            (-0.0f32).to_bits(),
            "-0.0 passes through bit-identically"
        );
        assert!(
            lower[1].is_nan(),
            "NaN passes through unclamped (fail-open), got {}",
            lower[1]
        );
        assert_eq!(
            lower[2].to_bits(),
            (-0.0f32 + super::REPLAY_ALPHA_TRUST_REGION).to_bits(),
            "overshoot from a -0.0 anchor lands exactly at anchor + 0.05"
        );
        assert!(upper.iter().all(|v| v.to_bits() == 0.5f32.to_bits()));
    }

    #[test]
    fn terminal_bound_only_requires_dead_returned_state() {
        assert!(!DagAlphaLoopResultUse::BoundsOnly.terminal_bound_only(false));
        assert!(DagAlphaLoopResultUse::BoundsOnly.terminal_bound_only(true));
        assert!(!DagAlphaLoopResultUse::BoundsAndState.terminal_bound_only(false));
        assert!(!DagAlphaLoopResultUse::BoundsAndState.terminal_bound_only(true));
    }

    #[test]
    fn dedup_reuse_requires_compatible_step1_authority() {
        assert!(!can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::CganCompleteCrownIbp {
                all_demanded_targets_completed: false,
            },
            false
        ));
        assert!(can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::CganCompleteCrownIbp {
                all_demanded_targets_completed: true,
            },
            true
        ));
        assert!(!can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::CganSparseTargetComplete {
                selected_target_completed: false,
            },
            false
        ));
        assert!(can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::CganSparseTargetComplete {
                selected_target_completed: true,
            },
            true
        ));
        assert!(can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::CrownIbp,
            false
        ));
        assert!(!can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::CrownIbp,
            true
        ));
        assert!(can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::ForwardLinear,
            true
        ));
        assert!(!can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::ForwardLinear,
            false
        ));
        assert!(!can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::Ibp,
            false
        ));
        assert!(!can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::Ibp,
            true
        ));
    }

    #[test]
    fn alpha_refresh_fraction_defaults_for_absent_or_invalid_values() {
        let fallback = AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION;
        assert_eq!(parse_alpha_refresh_fraction(None, fallback), fallback);
        for raw in [
            "",
            "not-a-number",
            "NaN",
            "inf",
            "-inf",
            "0",
            "-0.5",
            "0.009",
            "1.001",
        ] {
            assert_eq!(
                parse_alpha_refresh_fraction(Some(raw), fallback),
                fallback,
                "raw={raw:?}"
            );
        }

        assert_eq!(
            parse_alpha_refresh_fraction(None, 0.125),
            0.125,
            "an absent environment override must preserve the typed config"
        );
        assert_eq!(
            parse_alpha_refresh_fraction(Some("invalid"), 0.125),
            0.125,
            "a malformed environment override must fail closed to typed config"
        );
    }

    #[test]
    fn alpha_refresh_fraction_accepts_in_range_values_and_boundaries() {
        for (raw, expected) in [
            ("0.01", AlphaCrownConfig::MIN_REFERENCE_REFRESH_FRACTION),
            (" 0.125 ", 0.125),
            ("0.25", AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION),
            ("1", 1.0),
        ] {
            assert_eq!(
                parse_alpha_refresh_fraction(
                    Some(raw),
                    AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION,
                ),
                expected,
                "raw={raw:?}"
            );
        }
    }

    #[test]
    fn alpha_refresh_budget_is_cumulative_and_saturates_on_exhaustion() {
        let mut budget = None;
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut budget,
                Some(Duration::from_secs(100)),
                AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION,
                None,
            ),
            Some(Duration::from_secs(25))
        );

        debit_alpha_refresh_budget(&mut budget, Duration::from_secs(10));
        // A fresh 25%-of-remaining envelope would grant 22.5s here. The
        // cumulative policy exposes only the 15s left in the original pool.
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut budget,
                Some(Duration::from_secs(90)),
                AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION,
                None,
            ),
            Some(Duration::from_secs(15))
        );

        // The global verifier deadline remains an independent hard ceiling.
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut budget,
                Some(Duration::from_secs(4)),
                AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION,
                None,
            ),
            Some(Duration::from_secs(4))
        );

        debit_alpha_refresh_budget(&mut budget, Duration::from_secs(20));
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut budget,
                Some(Duration::from_secs(70)),
                AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION,
                None,
            ),
            Some(Duration::ZERO),
            "collector overrun must exhaust rather than replenish the pool"
        );
    }

    #[test]
    fn alpha_refresh_absolute_cap_clamps_long_budgets_and_preserves_short_ones() {
        let fraction = AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION;
        let maximum = Some(Duration::from_secs(12));

        let mut long_budget = None;
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut long_budget,
                Some(Duration::from_mins(15)),
                fraction,
                maximum,
            ),
            Some(Duration::from_secs(12)),
            "a long official budget must not scale the aggregate refresh past its ceiling"
        );
        debit_alpha_refresh_budget(&mut long_budget, Duration::from_secs(5));
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut long_budget,
                Some(Duration::from_secs(880)),
                fraction,
                maximum,
            ),
            Some(Duration::from_secs(7)),
            "later refreshes must share the original absolute cap instead of replenishing it"
        );

        let mut short_budget = None;
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut short_budget,
                Some(Duration::from_secs(40)),
                fraction,
                maximum,
            ),
            Some(Duration::from_secs(10)),
            "the fractional allowance must remain authoritative below the ceiling"
        );

        let mut cap_only = None;
        assert_eq!(
            cumulative_alpha_refresh_allowance(&mut cap_only, None, fraction, maximum),
            maximum,
            "an explicit cap must bound callers without a global deadline"
        );

        let mut unbounded = None;
        assert_eq!(
            cumulative_alpha_refresh_allowance(&mut unbounded, None, fraction, None),
            None,
            "default no-deadline callers must retain the historical unbounded path"
        );

        let mut disabled = None;
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut disabled,
                Some(Duration::from_mins(15)),
                fraction,
                Some(Duration::ZERO),
            ),
            Some(Duration::ZERO),
            "an explicit zero ceiling must disable refresh scheduling"
        );
    }

    #[test]
    fn exhausted_refresh_budget_does_not_enable_state_discard_shortcuts() {
        let mut budget = None;
        let allowance = cumulative_alpha_refresh_allowance(
            &mut budget,
            Some(Duration::from_secs(8)),
            AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION,
            None,
        )
        .expect("finite global deadline produces a finite allowance");
        debit_alpha_refresh_budget(&mut budget, allowance);
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut budget,
                Some(Duration::from_secs(6)),
                AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION,
                None,
            ),
            Some(Duration::ZERO)
        );

        // Exhausting a schedule-only reference-refresh pool must not make the
        // returned optimizer state dead. Collection still retains its terminal
        // update for immediate BaB re-evaluation.
        assert!(
            !DagAlphaLoopResultUse::BoundsAndState.terminal_bound_only(true),
            "state-returning DAG collection must continue to fail closed"
        );
    }

    #[test]
    fn root_alpha_margin_gate_arms_only_on_exact_one() {
        // The environment override is exact: only `1` arms it, while any other
        // present value kills the typed preset default. The matrix below also
        // covers the absent-override preset path.
        assert!(parse_root_alpha_margin(Some("1")));
        for raw in ["0", "true", "on", "yes", " 1", "1 ", "", "01", "2", "-1"] {
            assert!(
                !parse_root_alpha_margin(Some(raw)),
                "{raw:?} must not arm #root-alpha-margin"
            );
        }
        assert!(!parse_root_alpha_margin(None), "unset must not arm");

        assert!(
            root_alpha_margin_enabled_from(None, true),
            "an absent env override must preserve the typed preset default"
        );
        assert!(root_alpha_margin_enabled_from(
            Some(std::ffi::OsStr::new("1")),
            false,
        ));
        for raw in ["0", "true", " 1", ""] {
            assert!(
                !root_alpha_margin_enabled_from(Some(std::ffi::OsStr::new(raw)), true),
                "a present malformed override must kill a preset-enabled gate"
            );
        }

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let non_unicode = std::ffi::OsStr::from_bytes(&[0xff]);
            assert!(
                !root_alpha_margin_enabled_from(Some(non_unicode), true),
                "a present non-Unicode override must kill rather than reveal the preset default"
            );
        }
    }

    #[test]
    fn alpha_zero_yield_env_wins_both_directions_over_the_preset_default() {
        // Both absent: off, byte-identical.
        assert_eq!(alpha_zero_yield_frac_from(None, None), None);
        // Preset default applies when the env var is absent.
        assert_eq!(alpha_zero_yield_frac_from(None, Some(0.25)), Some(0.25));
        // An out-of-range preset value fails closed rather than arming.
        assert_eq!(alpha_zero_yield_frac_from(None, Some(0.9)), None);
        assert_eq!(alpha_zero_yield_frac_from(None, Some(0.0)), None);
        assert_eq!(alpha_zero_yield_frac_from(None, Some(f64::NAN)), None);
        // A present valid env value arms over a silent preset.
        assert_eq!(
            alpha_zero_yield_frac_from(Some(std::ffi::OsStr::new("0.5")), None),
            Some(0.5)
        );
        // A present valid env value REPLACES the preset value.
        assert_eq!(
            alpha_zero_yield_frac_from(Some(std::ffi::OsStr::new("0.5")), Some(0.25)),
            Some(0.5)
        );
        // A present invalid env value is a kill switch for a preset-armed
        // fraction — never a fall-through to the preset.
        for raw in ["0", "0.9", "1.0", "-0.1", "off", "", "nan", "inf"] {
            assert_eq!(
                alpha_zero_yield_frac_from(Some(std::ffi::OsStr::new(raw)), Some(0.25)),
                None,
                "{raw:?} must disarm the preset default, not reveal it"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn alpha_zero_yield_non_utf8_env_presence_kills_the_preset_default() {
        use std::os::unix::ffi::OsStrExt;

        let non_utf8 = std::ffi::OsStr::from_bytes(&[0xff]);
        assert_eq!(
            alpha_zero_yield_frac_from(Some(non_utf8), Some(0.25)),
            None,
            "a present non-UTF-8 override is invalid and must suppress, not reveal, the preset"
        );
    }

    #[test]
    fn root_alpha_margin_preset_default_reaches_the_operative_spec_gate() {
        let ascent = AlphaSpecAscent::new(vec![AlphaSpecEarlyExit {
            objective: vec![1.0],
            threshold: 0.0,
            verify_upper_bound: false,
        }])
        .expect("valid one-row spec");
        let config = AlphaCrownConfig {
            root_alpha_margin: true,
            spec_ascent: Some(ascent),
            ..AlphaCrownConfig::default()
        };

        let (enabled, spec_ascent) = root_alpha_margin_state_from(&config, None);
        assert!(enabled, "an absent env override must preserve preset=true");
        assert!(
            spec_ascent.is_some(),
            "the operative DAG loop must retain the preset-delivered spec objective"
        );

        let (enabled, spec_ascent) =
            root_alpha_margin_state_from(&config, Some(std::ffi::OsStr::new("0")));
        assert!(
            !enabled,
            "a present env kill switch must override the preset"
        );
        assert!(
            spec_ascent.is_none(),
            "the kill switch must restore the legacy no-ranking path"
        );
    }

    #[test]
    fn root_alpha_margin_gradient_gate_is_exact_and_not_read_without_parent() {
        assert!(parse_root_alpha_margin_gradient(Some("1")));
        for raw in ["0", "true", "on", "yes", " 1", "1 ", "", "01", "2", "-1"] {
            assert!(
                !parse_root_alpha_margin_gradient(Some(raw)),
                "{raw:?} must not arm the margin-gradient child"
            );
        }
        assert!(!parse_root_alpha_margin_gradient(None));

        let read = Cell::new(false);
        assert!(!root_alpha_margin_gradient_enabled_if(false, || {
            read.set(true);
            Some("1".to_string())
        }));
        assert!(
            !read.get(),
            "the child environment variable must not be read outside its parent gate"
        );
        assert!(root_alpha_margin_gradient_enabled_if(true, || {
            Some("1".to_string())
        }));
    }

    #[test]
    fn eligible_margin_gradient_retains_the_resident_local_rule_cache() {
        assert!(!retain_warmup_iter_cache(false, false));
        assert!(retain_warmup_iter_cache(true, false));
        assert!(retain_warmup_iter_cache(false, true));
        assert!(retain_warmup_iter_cache(true, true));
    }

    #[test]
    fn margin_gradient_conflicting_alpha_policies_are_explicitly_ineligible() {
        let eligibility = |multiobj, root_true| {
            margin_gradient_eligibility(true, true, true, true, multiobj, root_true)
        };
        assert_eq!(
            eligibility(true, false),
            MarginGradientEligibility::ConflictMultiobjJointAlpha
        );
        assert_eq!(
            eligibility(false, true),
            MarginGradientEligibility::ConflictRootAlphaTrue
        );
        assert_eq!(
            eligibility(true, true),
            MarginGradientEligibility::ConflictBothAlphaPolicies
        );
        assert_eq!(
            eligibility(false, false),
            MarginGradientEligibility::Eligible
        );
        assert_eq!(
            MarginGradientEligibility::ConflictMultiobjJointAlpha.reason(),
            "conflict_NY_MULTIOBJ_JOINT_ALPHA"
        );
        assert_eq!(
            MarginGradientEligibility::ConflictRootAlphaTrue.reason(),
            "conflict_NY_ROOT_ALPHA_TRUE"
        );
    }

    /// The divergence bail-out arms only on an exact `"1"`, and its threshold
    /// must sit far above ordinary ascent noise and far below the measured
    /// blow-up.
    #[test]
    fn alpha_divergence_bail_gate_and_threshold() {
        assert!(parse_alpha_divergence_bail(Some("1")));
        for declined in [None, Some(""), Some("0"), Some("true"), Some(" 1")] {
            assert!(
                !parse_alpha_divergence_bail(declined),
                "{declined:?} must not arm the divergence bail"
            );
        }
        // The measured case: best -1989.90, iterate -2.15e23 ⇒ diverged.
        let best: f64 = -1_989.896_240;
        let exploded: f64 = -2.149_872_7e23;
        assert!((best - exploded) > ALPHA_DIVERGENCE_FACTOR * best.abs().max(1.0));
        // An ordinary worse-but-comparable iterate must NOT trip it: the guard
        // is for divergence, not for the wrong-objective regressions that
        // #root-alpha-margin repairs.
        for ordinary in [best * 2.0, best * 100.0, best - 1.0e6] {
            assert!(
                (best - ordinary) <= ALPHA_DIVERGENCE_FACTOR * best.abs().max(1.0),
                "ordinary regression {ordinary:e} must not be treated as divergence"
            );
        }
        // A near-zero best must not make the guard hair-trigger.
        let tiny: f64 = -1e-9;
        assert!((tiny - (-0.5)) <= ALPHA_DIVERGENCE_FACTOR * tiny.abs().max(1.0));
    }

    /// The hinge sub-gate is subordinate: it can only arm when the gradient
    /// child gate is already on, and only on an exact `"1"`.
    #[test]
    fn hinge_steering_gate_is_subordinate_and_exact() {
        assert!(parse_root_alpha_margin_hinge(Some("1")));
        for declined in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some(" 1"),
            Some("1 "),
        ] {
            assert!(
                !parse_root_alpha_margin_hinge(declined),
                "{declined:?} must not arm hinge steering"
            );
        }
        // Parent (gradient) off ⇒ child can never arm, whatever the env says.
        assert!(!root_alpha_margin_hinge_enabled(false));
    }

    /// The hinge subgradient is the SUM of every violated row's objective, not
    /// the single worst one — that is what makes steering agree with the hinge
    /// score selection is judged by.
    #[test]
    fn hinge_margin_objective_sums_every_violated_row() {
        let ascent = AlphaSpecAscent::new(vec![
            AlphaSpecEarlyExit {
                objective: vec![1.0, 0.0, 0.0],
                threshold: 0.0,
                verify_upper_bound: false,
            },
            AlphaSpecEarlyExit {
                objective: vec![0.0, 1.0, 0.0],
                threshold: 0.0,
                verify_upper_bound: false,
            },
            AlphaSpecEarlyExit {
                objective: vec![0.0, 0.0, 1.0],
                threshold: 0.0,
                verify_upper_bound: false,
            },
        ])
        .expect("valid rows");
        // Rows 1 and 2 are violated (lower −3, −1); row 0 is satisfied (+2).
        let lo = [2.0, -3.0, -1.0];
        let hi = [3.0, -2.0, 0.0];
        let single = binding_margin_lower_objective(&ascent, &lo, &hi).expect("binding");
        let hinge = hinge_margin_lower_objective(&ascent, &lo, &hi).expect("hinge");

        // Single-row steering points at row 1 alone.
        assert_eq!(single.lower_objective, vec![0.0, 1.0, 0.0]);
        // Hinge steering points at rows 1 AND 2, and NOT at the satisfied row 0
        // — "only unproven rows contribute" is the whole point of the hinge.
        assert_eq!(hinge.lower_objective, vec![0.0, 1.0, 1.0]);
        // Telemetry still reports the worst contributing row, so existing logs
        // keep their meaning.
        assert_eq!(hinge.row_index, single.row_index);
        assert_eq!(hinge.slack, single.slack);
    }

    /// Upper-verification rows are sign-flipped before summing, matching the
    /// single-row convention, so mixed-direction conjunctions stay correct.
    #[test]
    fn hinge_margin_objective_sign_flips_upper_rows_before_summing() {
        let ascent = AlphaSpecAscent::new(vec![
            AlphaSpecEarlyExit {
                objective: vec![1.0, 0.0],
                threshold: 0.0,
                verify_upper_bound: false,
            },
            AlphaSpecEarlyExit {
                objective: vec![0.0, 1.0],
                threshold: 0.0,
                verify_upper_bound: true,
            },
        ])
        .expect("valid rows");
        let hinge = hinge_margin_lower_objective(&ascent, &[-1.0, 1.0], &[0.0, 2.0]);
        if let Some(h) = hinge {
            // Row 0 contributes +1 on slot 0; an upper row contributes NEGATED.
            assert_eq!(h.lower_objective[0], 1.0);
            assert!(
                h.lower_objective[1] <= 0.0,
                "upper-verification rows must enter the sum sign-flipped, got {}",
                h.lower_objective[1]
            );
        }
    }

    /// A fully-verified set has no violated rows, so there is no hinge
    /// subgradient to follow — refuse rather than return a zero direction.
    #[test]
    fn hinge_margin_objective_refuses_when_every_row_is_verified() {
        let ascent = AlphaSpecAscent::new(vec![AlphaSpecEarlyExit {
            objective: vec![1.0],
            threshold: 0.0,
            verify_upper_bound: false,
        }])
        .expect("valid rows");
        assert!(
            hinge_margin_lower_objective(&ascent, &[2.0], &[3.0]).is_none(),
            "no violated row ⇒ no direction"
        );
    }

    #[test]
    fn binding_margin_objective_selects_worst_unverified_row() {
        let ascent = AlphaSpecAscent::new(vec![
            AlphaSpecEarlyExit {
                objective: vec![1.0, 0.0, 0.0],
                threshold: 0.0,
                verify_upper_bound: false,
            },
            AlphaSpecEarlyExit {
                objective: vec![0.0, 1.0, 0.0],
                threshold: 0.0,
                verify_upper_bound: false,
            },
            AlphaSpecEarlyExit {
                objective: vec![0.0, 0.0, 1.0],
                threshold: 0.0,
                verify_upper_bound: false,
            },
        ])
        .expect("valid rows");
        let binding =
            binding_margin_lower_objective(&ascent, &[2.0, -3.0, -1.0], &[3.0, -2.0, 0.0])
                .expect("one binding row");
        assert_eq!(binding.row_index, 1);
        assert!(
            binding.slack <= -3.0 && binding.slack > -3.000_001,
            "directed projection rounds the lower slack outward"
        );
        assert_eq!(binding.lower_objective, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn binding_upper_verification_row_is_sign_flipped_to_lower_objective() {
        let ascent = AlphaSpecAscent::new(vec![AlphaSpecEarlyExit {
            objective: vec![2.0, -1.0],
            threshold: 0.0,
            verify_upper_bound: true,
        }])
        .expect("valid row");
        let binding = binding_margin_lower_objective(&ascent, &[0.0, 0.0], &[1.0, 1.0])
            .expect("upper row is unresolved");
        assert_eq!(binding.row_index, 0);
        assert!(
            binding.slack <= -2.0 && binding.slack > -2.000_001,
            "directed projection rounds the upper slack outward"
        );
        assert_eq!(binding.lower_objective, vec![-2.0, 1.0]);
    }

    #[test]
    fn binding_margin_objective_refuses_partial_or_fully_verified_sets() {
        let ascent = AlphaSpecAscent::new(vec![AlphaSpecEarlyExit {
            objective: vec![1.0, -1.0],
            threshold: 0.0,
            verify_upper_bound: false,
        }])
        .expect("valid row");
        assert!(
            binding_margin_lower_objective(&ascent, &[2.0, 0.0], &[3.0, 1.0]).is_none(),
            "a fully verified set needs no update"
        );
        assert!(
            binding_margin_lower_objective(&ascent, &[0.0], &[1.0]).is_none(),
            "shape mismatch must refuse rather than optimize a partial conjunction"
        );
    }
}
