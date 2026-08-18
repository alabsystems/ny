// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::gpu_suffix::{try_finish_target_gpu_suffix_with_pending_input, GpuSuffixPlan};
use super::target_backward_patches::{
    initial_target_crown_bounds_with_override, resolve_preactivation, target_allows_patches_start,
    try_patches_residual_target_step, try_patches_target_step_core,
};
use super::*;
use crate::bounds::patches::PatchesMaterializationPurpose;
use crate::bounds::LinearBounds;
use crate::network::core::apply_dense_backward_dispatch_result_with_deadline;
use crate::network::CrownMergeAccumulator;
use ndarray::{Array1, Array2, Ix1};
use ny_tensor::{next_down_f32, next_up_f32};

/// Hard row cap for the root-only output-conditioned two-seed evaluator.
///
/// The intended root-only caller selects at most eight dense-head coordinates.
/// Keeping the cap here as well makes the low-level evaluator fail closed if a
/// future caller bypasses that planner.
const OUTPUT_CONDITIONED_TWO_SEED_MAX_ROWS: usize = ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS;

/// A second CROWN frontier that is merged with the target identity before the
/// walk traverses the target's ancestors.
///
/// This is deliberately private to the target-backward core. In particular,
/// none of the single-seed GPU suffix shortcuts may consume it before the two
/// frontiers meet.
struct AdditionalTargetBackwardSeed<'a> {
    node_name: &'a str,
    bounds: CrownBounds,
}

/// Exact dense topology admitted for an output-conditioned additional seed.
///
/// This is intentionally narrower than [`Layer::propagates_coeff_err`]. The
/// target-backward loop has custom branches that bypass the canonical
/// `dispatch_backward_layer` coefficient-error safety net, so a broad layer
/// capability bit is not sufficient proof for this path.
///
/// The admitted CIFAR ResNet/toy subset has been checked as follows:
///
/// - `Linear`, dense `Conv2d`, and `BatchNorm` propagate incoming coefficient
///   error and attach fresh multiplication/accumulation error;
/// - fixed-slope `ReLU` uses the error-aware dense activation composition,
///   then eagerly folds its resulting error over the pre-activation box;
/// - dense `AveragePool` attaches a fresh gamma error (incoming error is
///   discharged before dispatch by the additional-seed CPU guard below);
/// - `Add` is an exact binary split and the merge accumulator certifies later
///   f32 accumulation;
/// - `Flatten` and `Reshape` are exact pass-through relations.
///
/// In particular, this refuses Mul/Div/Where, alpha-ReLU, and the custom alpha
/// Sigmoid/Tanh/Sqrt/Reciprocal branches. The alpha-ReLU implementation has
/// not yet been audited for the additional seed's certified error channel;
/// the other custom branches can reconstruct A/b without preserving it.
/// Everything else remains refused until it has an equally explicit audit.
fn output_conditioned_additional_seed_node_is_audited(layer: &Layer, input_count: usize) -> bool {
    match layer {
        Layer::Linear(_)
        | Layer::Conv2d(_)
        | Layer::BatchNorm(_)
        | Layer::ReLU(_)
        | Layer::AveragePool(_)
        | Layer::Flatten(_)
        | Layer::Reshape(_) => input_count == 1,
        Layer::Add(_) => input_count == 2,
        _ => false,
    }
}

fn output_conditioned_additional_seed_refusal_reason(
    layer: &Layer,
    node_name: &str,
    alpha_state: Option<&GraphAlphaState>,
) -> &'static str {
    match layer {
        Layer::MulBinary(_) => "MulBinary does not preserve incoming coefficient error",
        Layer::Div(_) => "Div does not preserve incoming coefficient error",
        Layer::Where(_) => "Where column routing does not preserve coefficient error",
        Layer::ReLU(_)
            if alpha_state
                .and_then(|state| state.alpha(node_name))
                .is_some() =>
        {
            "alpha-ReLU backward is not admitted for an output-conditioned additional seed"
        }
        Layer::Sigmoid(_) | Layer::Tanh(_)
            if alpha_state
                .and_then(|state| state.monotone_s_shaped_alpha(node_name))
                .is_some() =>
        {
            "custom-alpha S-shaped backward does not preserve coefficient error"
        }
        Layer::Sqrt(_)
            if alpha_state
                .and_then(|state| state.sqrt_alpha(node_name))
                .is_some() =>
        {
            "custom-alpha Sqrt backward does not preserve coefficient error"
        }
        Layer::Reciprocal(_)
            if alpha_state
                .and_then(|state| state.reciprocal_alpha(node_name))
                .is_some() =>
        {
            "custom-alpha Reciprocal backward does not preserve coefficient error"
        }
        _ => "layer/topology is outside the audited additional-seed dense subset",
    }
}

fn output_conditioned_cpu_error_box_is_usable(
    bounds: &BoundedTensor,
    expected_width: usize,
) -> bool {
    bounds.len() == expected_width
        && !bounds.lower().is_empty()
        && bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .all(|(&lower, &upper)| lower.is_finite() && upper.is_finite() && lower <= upper)
}

/// Discharge an additional-seed CPU frontier's incoming coefficient error over
/// the box for the value its columns multiply: this CURRENT node's output.
///
/// CROWN is preferred when it is a usable finite enclosure; IBP is the
/// fail-closed fallback. `fold_coeff_err_over_box_eager` deliberately retains
/// rows whose penalty is non-finite, so publication requires the error channel
/// to be completely absent afterwards.
fn fold_output_conditioned_additional_seed_error_for_cpu(
    node_name: &str,
    node_lb: &mut LinearBounds,
    crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
    ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
) -> Result<()> {
    if node_lb.has_coeff_err() {
        let expected_width = node_lb.num_inputs();
        let current_node_box = crown_bounds
            .get(node_name)
            .filter(|bounds| output_conditioned_cpu_error_box_is_usable(bounds, expected_width))
            .or_else(|| {
                ibp_bounds.get(node_name).filter(|bounds| {
                    output_conditioned_cpu_error_box_is_usable(bounds, expected_width)
                })
            })
            .ok_or_else(|| {
                NyError::SoundnessRefusal(format!(
                    "output-conditioned additional-seed CPU backward at '{node_name}' requires a \
                     finite ordered current-node output box of width {expected_width}"
                ))
            })?;
        node_lb.fold_coeff_err_over_box_eager(current_node_box);
    }

    if node_lb.has_coeff_err()
        || node_lb
            .lower_a()
            .iter()
            .chain(node_lb.upper_a().iter())
            .chain(node_lb.lower_b().iter())
            .chain(node_lb.upper_b().iter())
            .any(|value| !value.is_finite())
    {
        return Err(NyError::SoundnessRefusal(format!(
            "output-conditioned additional-seed CPU backward at '{node_name}' could not discharge \
             coefficient error into finite A/b"
        )));
    }
    Ok(())
}

/// Round one exact-f64 seed coefficient to f32 and return the certified
/// absolute storage error. A product of two binary32 values is exact in f64.
fn output_conditioned_seed_coeff(value: f64) -> Result<(f32, f32)> {
    if !value.is_finite() {
        return Err(NyError::InvalidSpec(
            "output-conditioned seed coefficient is non-finite".to_string(),
        ));
    }
    let stored = value as f32;
    if !stored.is_finite() {
        return Err(NyError::InvalidSpec(
            "output-conditioned seed coefficient overflows f32".to_string(),
        ));
    }
    let gap = (value - stored as f64).abs();
    let error = if gap == 0.0 {
        0.0
    } else {
        let rounded = gap as f32;
        if !rounded.is_finite() {
            return Err(NyError::InvalidSpec(
                "output-conditioned seed error overflows f32".to_string(),
            ));
        }
        next_up_f32(rounded)
    };
    Ok((stored, error))
}

fn output_conditioned_seed_bias(value: f64, lower: bool) -> Result<f32> {
    if !value.is_finite() {
        return Err(NyError::InvalidSpec(
            "output-conditioned seed bias is non-finite".to_string(),
        ));
    }
    let stored = value as f32;
    if !stored.is_finite() {
        return Err(NyError::InvalidSpec(
            "output-conditioned seed bias overflows f32".to_string(),
        ));
    }
    Ok(if lower && stored as f64 > value {
        next_down_f32(stored)
    } else if !lower && (stored as f64) < value {
        next_up_f32(stored)
    } else {
        stored
    })
}

/// Build the output-premise half of the two-node seed.
///
/// For the active row `s(x) = c^T f(x) - threshold` and non-negative
/// multipliers, row `j` represents
///
/// - lower: `+gamma_lower[j] * s(x)`;
/// - upper: `-gamma_upper[j] * s(x)`.
///
/// The target identity is seeded separately. Once both frontiers merge, the
/// ordinary sign-aware backward therefore bounds `z_j + gamma*s` from below
/// and `z_j - gamma*s` from above.
fn build_output_conditioned_output_seed(
    objective_row: &[f32],
    threshold: f32,
    gammas_lower: &[f32],
    gammas_upper: &[f32],
) -> Result<CrownBounds> {
    if objective_row.is_empty()
        || !threshold.is_finite()
        || gammas_lower.is_empty()
        || gammas_lower.len() != gammas_upper.len()
        || objective_row.iter().any(|value| !value.is_finite())
        || gammas_lower
            .iter()
            .chain(gammas_upper)
            .any(|gamma| !gamma.is_finite() || *gamma < 0.0)
    {
        return Err(NyError::InvalidSpec(
            "malformed output-conditioned output seed".to_string(),
        ));
    }

    let rows = gammas_lower.len();
    let cols = objective_row.len();
    let mut lower_a = Array2::<f32>::zeros((rows, cols));
    let mut upper_a = Array2::<f32>::zeros((rows, cols));
    let mut lower_a_err = Array2::<f32>::zeros((rows, cols));
    let mut upper_a_err = Array2::<f32>::zeros((rows, cols));
    let mut lower_b = Array1::<f32>::zeros(rows);
    let mut upper_b = Array1::<f32>::zeros(rows);
    let mut any_error = false;

    for row in 0..rows {
        let gamma_lower = gammas_lower[row] as f64;
        let gamma_upper = gammas_upper[row] as f64;
        for (col, &coefficient) in objective_row.iter().enumerate() {
            let (stored_lower, error_lower) =
                output_conditioned_seed_coeff(gamma_lower * coefficient as f64)?;
            let (stored_upper, error_upper) =
                output_conditioned_seed_coeff(-gamma_upper * coefficient as f64)?;
            lower_a[[row, col]] = stored_lower;
            upper_a[[row, col]] = stored_upper;
            lower_a_err[[row, col]] = error_lower;
            upper_a_err[[row, col]] = error_upper;
            any_error |= error_lower != 0.0 || error_upper != 0.0;
        }
        lower_b[row] = output_conditioned_seed_bias(-gamma_lower * threshold as f64, true)?;
        upper_b[row] = output_conditioned_seed_bias(gamma_upper * threshold as f64, false)?;
    }

    let linear = if any_error {
        LinearBounds::new_or_conservative_with_err(
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err,
            upper_a_err,
        )?
    } else {
        LinearBounds::new(lower_a, lower_b, upper_a, upper_b)?
    };
    Ok(CrownBounds::Dense(linear))
}

/// Environment knob for PRIMARY objective-chunking streaming (#patches-obj-chunk).
///
/// `NY_CROWN_OBJ_CHUNK = C`:
/// - unset or `0` (DEFAULT): DISABLED — the target backward runs in a single pass,
///   byte-for-byte identical to the pre-chunking behavior.
/// - `C > 0`: seed the backward pass in row-chunks of at most `C` objective rows,
///   concretize each chunk independently, and scatter into a pre-sized output.
///   Bound-equivalent to the single pass by row-independence (conv col2im scatter,
///   per-row error bounds, and per-row concretize are all row-local).
const CROWN_OBJ_CHUNK_ENV: &str = "NY_CROWN_OBJ_CHUNK";

/// Hard objective-row cap for deadline-bearing backward walks that cross a
/// ConvTranspose layer.
///
/// ConvTranspose2d's certified coefficient path currently contains several
/// whole-objective GEMMs (including the f64 recomputation) that cannot poll a
/// deadline internally.  Streaming at most 32 independent objective rows per
/// pass gives the existing between-pass deadline poll bounded granularity on
/// cGAN-class generators.  The exact sealed row-7 first target has 4,608 rows
/// and its unchunked backward exceeded 250 s; the cap reduces each unpolled
/// pass's objective-row workload by 144x.  This cap is used only when a caller
/// supplied a deadline and the target ancestry contains ConvTranspose;
/// no-deadline/default execution is byte-identical.
const DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS: usize = 32;

/// Kill switch for the measured-projection early abort (#chunk-abort).
///
/// `NY_NO_CHUNK_ABORT=1` restores the pre-fix behavior: a deadline-bearing
/// chunked target backward runs chunks until one of them crosses the per-node
/// deadline. Strict callers still discard the walk; an explicitly armed
/// CROWN-IBP collector can retain already completed rows over its certified
/// IBP seed. See [`chunk_projection_abort_enabled`].
const NO_CHUNK_ABORT_ENV: &str = "NY_NO_CHUNK_ABORT";

/// Minimum measured chunks before the projection is trusted (#chunk-abort).
///
/// One sample can be distorted by first-touch allocation/cache effects; two
/// give a stable mean for the identically-shaped remaining chunks.
const CHUNK_ABORT_MIN_SAMPLES: usize = 2;

/// Slack on the #chunk-abort projection.
///
/// The projection uses the LAST chunk's per-row rate; a wider next chunk
/// amortizes the ancestor-prefix re-walk further, so the rate is a slight
/// OVER-estimate while #chunk-grow is still widening. Requiring the projection
/// to exceed the remaining budget by this factor keeps a node that would in
/// fact have finished from being abandoned.
const CHUNK_ABORT_SLACK: f64 = 1.25;

/// Kill switch for adaptive objective-chunk widening (#chunk-grow).
///
/// `NY_NO_CHUNK_GROW=1` restores the fixed-width partition (every chunk exactly
/// `effective_target_chunk_size` rows).
const NO_CHUNK_GROW_ENV: &str = "NY_NO_CHUNK_GROW";
const NO_CHUNK_WAVE_PAR_ENV: &str = "NY_NO_CHUNK_WAVE_PAR";

/// Target wall time for ONE unpolled objective chunk (#chunk-grow).
///
/// The 32-row `DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS` cap exists because the
/// ConvTranspose coefficient path cannot poll a deadline internally, so one
/// chunk is the granularity of the deadline. 10 s of unpolled work keeps that
/// granularity fine against the per-node budgets in play (12 s adaptive-cap
/// floor, up to 600 s from remaining-budget or preset scaling) while cutting
/// the number of prefix re-walks by orders of magnitude.
const CHUNK_GROW_TARGET_SECS: f64 = 10.0;

/// Maximum per-step widening factor (#chunk-grow), so one mis-measured chunk
/// cannot jump straight to a huge unpolled pass.
const CHUNK_GROW_MAX_FACTOR: usize = 4;

/// Default-dark production gate for retaining fully committed objective rows
/// when a collector target walk reaches its deadline.
///
/// Exact `1` is the only enabling value. The low-level salvage mechanism stays
/// available independently for soundness tests, while scored runs must opt in
/// explicitly until repeated benchmark evidence justifies promotion.
pub(super) const PARTIAL_CROWN_DEADLINE_SALVAGE_ENV: &str = "NY_CROWN_DEADLINE_CHUNK_SALVAGE";

/// Immutable policy snapshot for one CROWN-IBP collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::network::graph_alpha) enum PartialCrownDeadlineSalvagePolicy {
    Disabled,
    EnabledByExactEnvironment,
}

impl PartialCrownDeadlineSalvagePolicy {
    pub(super) fn from_raw(raw: Option<&str>) -> Self {
        if raw == Some("1") {
            Self::EnabledByExactEnvironment
        } else {
            Self::Disabled
        }
    }

    pub(super) fn from_environment() -> Self {
        Self::from_raw(
            std::env::var(PARTIAL_CROWN_DEADLINE_SALVAGE_ENV)
                .ok()
                .as_deref(),
        )
    }

    pub(super) const fn is_enabled(self) -> bool {
        matches!(self, Self::EnabledByExactEnvironment)
    }
}

#[cfg(test)]
std::thread_local! {
    /// Deterministic, one-shot controller boundary used by deadline-salvage
    /// tests. Unlike a sleep/near-now deadline, this cannot race with CI load.
    static TEST_CHUNK_DEADLINE_AFTER_COMMITS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
struct TestChunkDeadlineGuard {
    previous: Option<usize>,
}

#[cfg(test)]
impl TestChunkDeadlineGuard {
    fn after_committed_chunks(completed_chunks: usize) -> Self {
        let previous =
            TEST_CHUNK_DEADLINE_AFTER_COMMITS.with(|slot| slot.replace(Some(completed_chunks)));
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for TestChunkDeadlineGuard {
    fn drop(&mut self) {
        TEST_CHUNK_DEADLINE_AFTER_COMMITS.with(|slot| slot.set(self.previous));
    }
}

#[cfg(test)]
fn injected_chunk_controller_deadline(completed_chunks: usize) -> Option<NyError> {
    TEST_CHUNK_DEADLINE_AFTER_COMMITS.with(|slot| {
        slot.get()
            .filter(|threshold| completed_chunks >= *threshold)
            .map(|threshold| {
                // One shot: later targets in the same collector run normally.
                slot.set(None);
                NyError::DeadlineExceeded(format!(
                    "deterministic test deadline after {threshold} committed objective chunks"
                ))
            })
    })
}

#[cfg(not(test))]
#[inline]
fn injected_chunk_controller_deadline(_completed_chunks: usize) -> Option<NyError> {
    None
}

#[inline]
fn chunk_post_join_now(
    _deadline: std::time::Instant,
    _completed_chunks: usize,
) -> std::time::Instant {
    std::time::Instant::now()
}

/// Whether adaptive objective-chunk widening is enabled (default: yes).
pub(in crate::network::graph_alpha) fn chunk_adaptive_growth_enabled() -> bool {
    !matches!(std::env::var(NO_CHUNK_GROW_ENV).ok().as_deref(), Some("1"))
}

/// Whether deadline-bearing objective chunks use the fixed-width wave driver.
pub(in crate::network::graph_alpha) fn chunk_wave_parallel_enabled() -> bool {
    !matches!(
        std::env::var(NO_CHUNK_WAVE_PAR_ENV).ok().as_deref(),
        Some("1")
    )
}

/// Whether the measured-projection early abort is enabled (default: yes).
///
/// A deadline-bearing chunked backward that cannot finish inside its per-node
/// budget should stop before spending the whole share. When the default-dark
/// salvage policy is armed, the collector preserves fully completed rows over
/// certified IBP and marks the target explicitly truncated; otherwise it and
/// strict callers retain the historical `DeadlineExceeded`.
/// Measured on cgan_2023
/// (cGAN_imgSz32_nCh_1 prop_1, 900 s official budget, preset loaded):
/// `ConvTranspose_7` burned its whole 150.002 s per-node cap and produced
/// nothing, inside a disjunctive precheck phase that owns 0.85 of the budget.
/// Projecting the finish from the measured chunk cost and aborting as soon as
/// the projection passes the deadline hands the unspent seconds back to the
/// remaining targets (and, through the phase ledger, to BaB) while retaining
/// every already certified row.
pub(in crate::network::graph_alpha) fn chunk_projection_abort_enabled() -> bool {
    !matches!(std::env::var(NO_CHUNK_ABORT_ENV).ok().as_deref(), Some("1"))
}

/// Hard aggregate heap cap for the opt-in target-to-input linear-certificate
/// API's owned dense payload.
///
/// The normal concrete target backward is intentionally unaffected. The
/// certificate API charges the simultaneously resident full assembly plus a
/// conservative upper bound on one row-chunk's relations/transients. Identity
/// rows stream until that SUM fits; a request is refused before allocation when
/// the assembly plus even one row cannot fit.
const TARGET_INPUT_LINEAR_AGGREGATE_MAX_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy)]
struct TargetInputLinearLimits {
    aggregate_bytes: usize,
}

impl TargetInputLinearLimits {
    const PRODUCTION: Self = Self {
        aggregate_bytes: TARGET_INPUT_LINEAR_AGGREGATE_MAX_BYTES,
    };
}

// Both payloads are existing by-value return types. Boxing the concrete arm
// would add a heap allocation to the default path solely for this opt-in API.
#[allow(clippy::large_enum_variant)]
enum TargetBackwardPassResult {
    NoInputContribution,
    Concrete(BoundedTensor),
    InputLinear(LinearBounds),
}

type CompletedCrownChunk = (usize, Vec<f32>, Vec<f32>);

/// Result of one collector-target CROWN walk.
///
/// Ordinary CROWN/alpha-CROWN callers accept only `Complete`. The
/// CROWN-IBP collector additionally accepts `DeadlineTruncated`: its bound is
/// assembled from the certified IBP target box, with only fully completed
/// objective chunks overwritten. Keeping the variants distinct prevents a
/// partially collected target from being mistaken for a complete CROWN result.
pub(in crate::network::graph_alpha) enum TargetCrownCollectionResult {
    Complete(BoundedTensor),
    DeadlineTruncated {
        bounds: BoundedTensor,
        completed_rows: usize,
        total_rows: usize,
        details: String,
    },
}

impl TargetCrownCollectionResult {
    fn into_strict_result(self) -> Result<BoundedTensor> {
        match self {
            Self::Complete(bounds) => Ok(bounds),
            Self::DeadlineTruncated { details, .. } => Err(NyError::DeadlineExceeded(details)),
        }
    }
}

/// Assemble completed objective chunks over a certified target box.
///
/// The output starts as the exact forward/IBP enclosure. A row is overwritten
/// only after its whole CROWN chunk has returned successfully and passed the
/// post-chunk deadline poll. A late parallel wave is therefore represented by
/// omitting the entire wave from `completed_chunks`; none of its in-flight rows
/// can leak into the result.
fn assemble_completed_crown_chunks(
    target_bounds: &BoundedTensor,
    target_contract: &GraphTargetShapeContract,
    target_dim: usize,
    completed_chunks: Vec<CompletedCrownChunk>,
    deadline_truncation: Option<String>,
) -> Result<TargetCrownCollectionResult> {
    let target_flat = target_bounds.flatten();
    if target_flat.len() != target_dim {
        return Err(NyError::ShapeMismatch {
            expected: vec![target_dim],
            got: vec![target_flat.len()],
        });
    }

    // Certified seed for every unfinished row. This is intentionally a copy of
    // the caller's forward/IBP box rather than a zero placeholder: returning
    // early with zero completed chunks must still publish a valid enclosure.
    let mut out_lower = target_flat.lower().iter().copied().collect::<Vec<_>>();
    let mut out_upper = target_flat.upper().iter().copied().collect::<Vec<_>>();
    let mut completed = vec![false; target_dim];

    for (r0, lower, upper) in completed_chunks {
        if lower.len() != upper.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![lower.len()],
                got: vec![upper.len()],
            });
        }
        let r1 = r0.checked_add(lower.len()).ok_or_else(|| {
            NyError::InvalidSpec("objective-chunk row range overflow".to_string())
        })?;
        if r1 > target_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![target_dim],
                got: vec![r1],
            });
        }
        if completed[r0..r1].iter().any(|done| *done) {
            return Err(NyError::InternalError(format!(
                "objective-chunk rows {r0}..{r1} were completed more than once"
            )));
        }

        out_lower[r0..r1].copy_from_slice(&lower);
        out_upper[r0..r1].copy_from_slice(&upper);
        completed[r0..r1].fill(true);
    }

    let completed_rows = completed.iter().filter(|done| **done).count();
    if deadline_truncation.is_none() && completed_rows != target_dim {
        return Err(NyError::InternalError(format!(
            "complete objective-chunk walk assembled only {completed_rows}/{target_dim} rows"
        )));
    }

    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[target_dim]), out_lower)
        .map_err(|e| NyError::InvalidSpec(format!("objective-chunk lower reshape: {e}")))?;
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[target_dim]), out_upper)
        .map_err(|e| NyError::InvalidSpec(format!("objective-chunk upper reshape: {e}")))?;
    let assembled = BoundedTensor::new_allow_infinite(lower, upper)?;
    let bounds =
        target_contract.restore_concrete(assembled, "Graph alpha-CROWN objective-chunk restore")?;

    Ok(match deadline_truncation {
        Some(details) => TargetCrownCollectionResult::DeadlineTruncated {
            bounds,
            completed_rows,
            total_rows: target_dim,
            details,
        },
        None => TargetCrownCollectionResult::Complete(bounds),
    })
}

/// Checked worst-case heap payload for the raw assembled input-linear
/// certificate:
/// `{lower,upper}_a`, `{lower,upper}_a_err`, and `{lower,upper}_b`.
fn target_input_linear_raw_bytes(target_dim: usize, input_dim: usize) -> Option<usize> {
    let coeff_entries = target_dim.checked_mul(input_dim)?;
    let coeff_bytes = coeff_entries
        .checked_mul(size_of::<f32>())?
        .checked_mul(4)?;
    let bias_bytes = target_dim.checked_mul(size_of::<f32>())?.checked_mul(2)?;
    coeff_bytes.checked_add(bias_bytes)
}

/// Full raw assembly plus the workspace that remains live while the public
/// wrapper discharges coefficient error over the supplied input box.
fn target_input_linear_fixed_bytes(target_dim: usize, input_dim: usize) -> Option<usize> {
    let raw_bytes = target_input_linear_raw_bytes(target_dim, input_dim)?;
    // `BoundedTensor::flatten` allocates both f32 endpoints. The error fold
    // additionally holds one f64 magnitude per input column.
    let flat_input_bytes = input_dim.checked_mul(size_of::<f32>())?.checked_mul(2)?;
    let fold_magnitude_bytes = input_dim.checked_mul(size_of::<f64>())?;
    raw_bytes
        .checked_add(flat_input_bytes)?
        .checked_add(fold_magnitude_bytes)
}

/// Conservative per-row charge for all relations owned by one capture chunk.
///
/// A relation can carry lower/upper A plus lower/upper coefficient error
/// (`4*f32` per column) and lower/upper bias (`2*f32` per row). Summing every
/// relevant node width bounds a DAG frontier even if every relation coexists.
///
/// The factor six covers the worst merge/conversion phase observed in
/// `CrownMergeAccumulator`: existing and incoming f32 relations, an f64
/// relation (including f64 coefficient-error matrices), two f64 roundoff
/// matrices, and one relation-equivalent reserve for downcast rows or Patches
/// conversion. The dense identity seed (`2*f32*target_dim`) is included
/// explicitly and the larger charge wins.
fn target_input_linear_chunk_row_bytes(
    target_dim: usize,
    relation_cols_sum: usize,
    relation_count: usize,
) -> Option<usize> {
    let relation_coeff = relation_cols_sum
        .checked_mul(size_of::<f32>())?
        .checked_mul(4)?;
    let relation_bias = relation_count
        .checked_mul(size_of::<f32>())?
        .checked_mul(2)?;
    let relation_peak = relation_coeff.checked_add(relation_bias)?.checked_mul(6)?;
    let identity_seed = crate::network::crown_memory::dense_pair_bytes(1, target_dim)?;
    Some(relation_peak.max(identity_seed))
}

/// Choose a deterministic identity-row chunk whose assembly + chunk payload
/// satisfies the hard aggregate cap and the existing deadline-bearing
/// ConvTranspose cap.
fn target_input_linear_chunk_rows(
    target_dim: usize,
    assembly_bytes: usize,
    chunk_row_bytes: usize,
    has_deadline: bool,
    has_conv_transpose: bool,
    aggregate_cap_bytes: usize,
) -> Result<usize> {
    if target_dim == 0 {
        return Err(NyError::InvalidSpec(
            "target input-linear CROWN requires a non-empty target".to_string(),
        ));
    }
    let one_row_peak = assembly_bytes.saturating_add(chunk_row_bytes);
    if one_row_peak > aggregate_cap_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: one_row_peak,
            budget_bytes: aggregate_cap_bytes,
            site: "graph-alpha target input-linear aggregate workspace",
        });
    }
    let available = aggregate_cap_bytes - assembly_bytes;
    let rows_by_memory = (available / chunk_row_bytes).max(1).min(target_dim);
    let requested = if rows_by_memory < target_dim {
        rows_by_memory
    } else {
        0
    };
    let effective = effective_target_chunk_size(requested, has_deadline, has_conv_transpose);
    Ok(if effective == 0 {
        target_dim
    } else {
        effective.min(target_dim)
    })
}

/// Parse the objective row-chunk size. Returns 0 (disabled) when unset, empty,
/// non-numeric, or explicitly 0 — preserving the single-pass default.
pub(in crate::network::graph_alpha) fn crown_obj_chunk_size() -> usize {
    std::env::var(CROWN_OBJ_CHUNK_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

/// Combine the caller/env chunk request with the deadline-safety cap.
/// A zero requested size means "no explicit chunking", but cannot disable the
/// safety cap when a deadline-bearing ConvTranspose walk needs it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::network::graph_alpha) struct ObjectiveChunkRoutePlan {
    /// Caller/env width before the deadline-bearing ConvTranspose cap.
    pub(in crate::network::graph_alpha) requested_rows: usize,
    /// Width of the first execution wave/pass after applying that cap.
    ///
    /// Deadline-bearing ConvTranspose walks start at at most 32 rows. The
    /// sequential driver may widen later chunks adaptively, while the default
    /// parallel-wave driver retains this partition. Pre-run scheduling can
    /// therefore use this only as a conservative initial-wave proxy, not as a
    /// claim about the final adaptive partition.
    pub(in crate::network::graph_alpha) effective_initial_rows: usize,
}

pub(in crate::network::graph_alpha) fn objective_chunk_route_plan(
    requested_rows: usize,
    has_deadline: bool,
    has_conv_transpose: bool,
) -> ObjectiveChunkRoutePlan {
    ObjectiveChunkRoutePlan {
        requested_rows,
        effective_initial_rows: if !has_deadline || !has_conv_transpose {
            requested_rows
        } else if requested_rows == 0 {
            DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS
        } else {
            requested_rows.min(DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS)
        },
    }
}

/// Exact fixed-wave grouping selected by the objective-chunk driver.
///
/// These values are derived by the same helper used at execution time. M1 may
/// model this route because its chunk width and wave count are invariant for
/// the duration of the backward. Sequential adaptive growth is deliberately
/// represented by a different route and is not eligible for pre-run weighting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::network::graph_alpha) struct ObjectiveChunkFixedWavePlan {
    pub(in crate::network::graph_alpha) chunk_rows: usize,
    pub(in crate::network::graph_alpha) chunk_count: usize,
    pub(in crate::network::graph_alpha) wave_size: usize,
    pub(in crate::network::graph_alpha) wave_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::network::graph_alpha) enum ObjectiveChunkDriverRoute {
    FixedWaves(ObjectiveChunkFixedWavePlan),
    AnchorParallel,
    Sequential { adaptive_growth: bool },
}

/// Central objective-chunk driver decision used by execution and M1 planning.
///
/// Keeping this pure makes every routing input explicit: deadline authority,
/// cut context, wave kill switch, IMB anchor mode, adaptive-growth switch,
/// Rayon worker count, and the memory-derived concurrency cap.
#[allow(clippy::too_many_arguments)]
pub(in crate::network::graph_alpha) fn objective_chunk_driver_route(
    target_rows: usize,
    chunk_rows: usize,
    chunk_ceiling: usize,
    has_deadline: bool,
    has_cut_context: bool,
    wave_parallel_enabled: bool,
    anchor_parallel_enabled: bool,
    adaptive_growth_enabled: bool,
    worker_count: usize,
) -> ObjectiveChunkDriverRoute {
    let chunk_rows = chunk_rows.max(1);
    let chunk_count =
        target_rows / chunk_rows + usize::from(!target_rows.is_multiple_of(chunk_rows));
    if !has_cut_context && has_deadline && chunk_count > 1 && wave_parallel_enabled {
        let memory_wave_cap = (chunk_ceiling / chunk_rows).max(1);
        let wave_size = worker_count.max(1).min(memory_wave_cap).max(1);
        let wave_count =
            chunk_count / wave_size + usize::from(!chunk_count.is_multiple_of(wave_size));
        ObjectiveChunkDriverRoute::FixedWaves(ObjectiveChunkFixedWavePlan {
            chunk_rows,
            chunk_count,
            wave_size,
            wave_count,
        })
    } else if anchor_parallel_enabled && !has_cut_context && !has_deadline {
        ObjectiveChunkDriverRoute::AnchorParallel
    } else {
        ObjectiveChunkDriverRoute::Sequential {
            adaptive_growth: adaptive_growth_enabled,
        }
    }
}

/// Resolve the live driver route from the same process state execution reads.
pub(in crate::network::graph_alpha) fn live_objective_chunk_driver_route(
    target_rows: usize,
    chunk_rows: usize,
    chunk_ceiling: usize,
    has_deadline: bool,
    has_cut_context: bool,
) -> ObjectiveChunkDriverRoute {
    objective_chunk_driver_route(
        target_rows,
        chunk_rows,
        chunk_ceiling,
        has_deadline,
        has_cut_context,
        chunk_wave_parallel_enabled(),
        crate::imb::anchor_chunk_parallel(),
        chunk_adaptive_growth_enabled(),
        rayon::current_num_threads(),
    )
}

/// Bind an M1 scheduling admission to the route execution actually selected.
///
/// The expected plan is present only when the budget numerator/denominator
/// inflated this target by its fixed-wave count. If any live routing input
/// changed after admission (worker count, kill switch, cut/deadline authority,
/// or memory cap), refuse this optional tightening before executing it. The
/// caller then retains its sound reference/IBP bound instead of spending a
/// slice derived from work that will not run.
fn enforce_expected_fixed_wave_plan(
    actual: ObjectiveChunkDriverRoute,
    expected: Option<ObjectiveChunkFixedWavePlan>,
    label: &str,
    target_node: &str,
) -> Result<ObjectiveChunkDriverRoute> {
    if let Some(expected) = expected {
        match actual {
            ObjectiveChunkDriverRoute::FixedWaves(actual) if actual == expected => {}
            _ => {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{label}: objective-chunk route changed after M1 admission for target \
                     '{target_node}' (expected {expected:?}, actual {actual:?})"
                )));
            }
        }
    }
    Ok(actual)
}

fn effective_target_chunk_size(
    requested: usize,
    has_deadline: bool,
    has_conv_transpose: bool,
) -> usize {
    objective_chunk_route_plan(requested, has_deadline, has_conv_transpose).effective_initial_rows
}

/// Environment knob for the CROWN-IBP sweep's backward-to-nearest-bounded-cut
/// (#crown-cut-segment).
///
/// `NY_CROWN_CUT_SEGMENT = N`:
/// - unset or `0` (DEFAULT): DISABLED — every per-target backward expands its
///   full prefix to the network input, byte-for-byte the prior behavior.
/// - `N > 0`: the CROWN-IBP collector designates every node whose topological
///   index is a multiple of `N` as a CUT node. A per-target backward that
///   reaches a cut node whose bounds THIS sweep already finalized concretizes
///   the accumulated linear relation against that node's bound-box — the same
///   directed-rounding concretization the input-box path uses
///   (`LinearBounds::concretize_sound`) — instead of expanding through the
///   node's prefix. SOUND: the swept box is a valid enclosure of the cut
///   node's reachable set, so a linear relation concretized over it encloses
///   the relation's value for every reachable input; the input box is just
///   the trivial cut. The result is generally LOOSER than the full-prefix
///   backward (the box drops inter-node correlations), never tighter than a
///   valid enclosure. The sweep cost drops from O(n²) prefix steps to
///   ~O(n·N). Only the CROWN-IBP collection sweep passes a cut context; the
///   verdict-shaped α-CROWN output backward always runs full depth.
pub(in crate::network::graph_alpha) const CROWN_CUT_SEGMENT_ENV: &str = "NY_CROWN_CUT_SEGMENT";

/// Parse the cut segment length from the environment. Returns 0 (disabled)
/// when unset, empty, non-numeric, or explicitly 0 — preserving the
/// full-prefix default.
pub(in crate::network::graph_alpha) fn crown_cut_segment_from_env() -> usize {
    parse_crown_cut_segment(std::env::var(CROWN_CUT_SEGMENT_ENV).ok().as_deref())
}

/// Cap on the per-block materialization budget for the blockwise Patches
/// final concretization (#patches-row-range). The effective block budget is
/// `min(cpu_crown_dense_budget_bytes(), this)`: even a user-raised
/// `NY_DENSE_BUDGET_MB` never materializes more than 1 GiB of dense rows at a
/// time on that path (the peak also carries the transient err accumulators,
/// see `PatchesLinearBounds::concretize_sound_chunked`).
const PATCHES_CONCRETIZE_MAX_BLOCK_BYTES: usize = 1 << 30;

/// #patches-row-range: mid-walk densify budget guard. Returns the structured
/// `CpuMemoryExceeded` when a Patches-carried relation's dense pair would
/// exceed `budget_bytes` (or overflows the byte estimate, which saturates to
/// `usize::MAX`); `None` when it fits — or when the estimate itself errors, so
/// the subsequent `into_dense()` surfaces the original (shape) error
/// unchanged. `CpuMemoryExceeded` is mapped to a sound fallback by every CROWN
/// caller (CROWN-IBP collector -> IBP for the target; alpha paths likewise),
/// which is strictly better than aborting the process on a TB-scale
/// allocation (VGG16: 3.2M x 150K rows ~ 1.9 TB per matrix).
fn patches_densify_over_budget(pb: &PatchesLinearBounds, budget_bytes: usize) -> Option<NyError> {
    match pb.dense_pair_bytes() {
        Ok(required) if required > budget_bytes => Some(NyError::CpuMemoryExceeded {
            required_bytes: required,
            budget_bytes,
            site: "graph-alpha target-backward mid-walk patches densify",
        }),
        _ => None,
    }
}

/// Prepare the optional GPU-resnet seed without cloning either carrier before
/// checked admission. Dense seeds are borrowed because the CPU fallback keeps
/// owning the source; Patches seeds materialize transactionally into `Owned`.
fn materialize_optional_resnet_seed(
    bounds: &CrownBounds,
    deadline: Option<std::time::Instant>,
) -> Result<std::borrow::Cow<'_, LinearBounds>> {
    if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "graph-alpha optional GPU-resnet seed deadline expired before admission".to_string(),
        ));
    }
    match bounds {
        CrownBounds::Dense(bounds) => Ok(std::borrow::Cow::Borrowed(bounds)),
        CrownBounds::Patches(bounds) => bounds
            .to_dense_with_deadline(deadline)
            .map(std::borrow::Cow::Owned),
    }
}

/// Move a concretized one-dimensional bound pair out without the deep copies
/// performed by `BoundedTensor::flatten_to_ix1`.
fn concrete_into_ix1(concrete: BoundedTensor, context: &str) -> Result<(Array1<f32>, Array1<f32>)> {
    let (lower, upper) = concrete.into_parts();
    let lower_shape = lower.shape().to_vec();
    let upper_shape = upper.shape().to_vec();
    let lower = lower.into_dimensionality::<Ix1>().map_err(|error| {
        NyError::InternalError(format!(
            "{context}: concretized lower bound must be one-dimensional (shape {lower_shape:?}): \
             {error}"
        ))
    })?;
    let upper = upper.into_dimensionality::<Ix1>().map_err(|error| {
        NyError::InternalError(format!(
            "{context}: concretized upper bound must be one-dimensional (shape {upper_shape:?}): \
             {error}"
        ))
    })?;
    Ok((lower, upper))
}

/// Pure parser for [`CROWN_CUT_SEGMENT_ENV`] (unit-testable without touching
/// process env). `None`/empty/non-numeric/`0` all mean disabled.
fn parse_crown_cut_segment(raw: Option<&str>) -> usize {
    raw.and_then(|r| r.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

/// Cut-node context for the CROWN-IBP sweep (#crown-cut-segment,
/// `NY_CROWN_CUT_SEGMENT`). Built once per collection when the gate is on;
/// `None` everywhere else (α-CROWN backward, gate off) keeps the backward walk
/// byte-identical to the pre-cut behavior.
pub(crate) struct CrownCutContext {
    /// Nodes designated as cuts (topological index ≡ 0 mod segment length).
    /// Coverage of every path is NOT required for soundness — the input box
    /// remains the ultimate cut; membership only short-circuits the walk.
    cut_nodes: std::collections::HashSet<String>,
    /// Nodes whose finalized bound came from CROWN rather than an IBP fallback
    /// (#cut-provenance-gate). Only these may actually serve as cuts.
    ///
    /// `RefCell` suffices for the same reason as `cuts_used`: the sweep and its
    /// backward walks are single-threaded, and the collector registers a node
    /// only after that node's own tightening has finished.
    tight_nodes: std::cell::RefCell<std::collections::HashSet<String>>,
    /// Number of cut concretizations performed (for the sweep info line).
    /// `Cell` suffices: the sweep and its backward walks are single-threaded.
    cuts_used: std::cell::Cell<usize>,
}

impl CrownCutContext {
    pub(in crate::network::graph_alpha) fn new(
        cut_nodes: std::collections::HashSet<String>,
    ) -> Self {
        Self {
            cut_nodes,
            tight_nodes: std::cell::RefCell::new(std::collections::HashSet::new()),
            cuts_used: std::cell::Cell::new(0),
        }
    }

    /// Register a node whose bound was produced by CROWN, making it eligible to
    /// serve as a cut (#cut-provenance-gate).
    pub(in crate::network::graph_alpha) fn mark_tight(&self, node_name: &str) {
        self.tight_nodes.borrow_mut().insert(node_name.to_string());
    }

    /// A designated cut is only USED when its own bound is CROWN-tight.
    ///
    /// WHY (measured, TinyYOLO / yolo_2023, 2026-07-28). Cutting concretizes the
    /// accumulated relation against the cut node's box, so the cut inherits that
    /// box's quality. Cutting at a node that itself fell back to IBP therefore
    /// injects an IBP-width box into the middle of an otherwise tight walk and
    /// destroys everything downstream of it.
    ///
    /// The uniform every-N-th placement made that failure mode load-bearing and
    /// invisible. Sweeping N: `SEGMENT=8` (cuts Add_8/Relu_16/Conv_24) and
    /// `SEGMENT=16` (cuts Relu_16) hold the root bound at ~`[-39087, 20812]`
    /// while cutting collection time 27%, but `SEGMENT=4` and `SEGMENT=12`
    /// COLLAPSE it back to `[-202134, 165731]`. The two collapsing arms are
    /// exactly the two that cut at `Conv_12` — whose own provenance on this
    /// instance is `ForwardFallback(PerNodeDeadlineExceeded)`, i.e. IBP. The
    /// distinguishing factor is not how many cuts, but whether a cut node's own
    /// bound is tight.
    ///
    /// Gating on provenance removes the sharp non-monotonicity in N: a node that
    /// would poison the walk is simply expanded as usual (the fail-open path
    /// that already exists for a cut node the map does not cover).
    ///
    /// Sound either way — cutting only ever LOOSENS a bound, so declining a cut
    /// cannot make one wrong.
    fn is_cut(&self, node_name: &str) -> bool {
        self.cut_nodes.contains(node_name) && self.tight_nodes.borrow().contains(node_name)
    }

    fn record_cut(&self) {
        self.cuts_used.set(self.cuts_used.get() + 1);
    }

    pub(in crate::network::graph_alpha) fn cuts_used(&self) -> usize {
        self.cuts_used.get()
    }
}

/// Whether every endpoint of the box is finite. A cut concretization over a
/// non-finite box would be vacuous (±inf rows); the walk expands through the
/// node instead (fail-open to the exact, tighter behavior).
fn box_is_finite(b: &BoundedTensor) -> bool {
    b.lower().iter().all(|v| v.is_finite()) && b.upper().iter().all(|v| v.is_finite())
}

impl GraphNetwork {
    /// Shared backward CROWN core for per-target CROWN-IBP and α-CROWN.
    /// `collector_patches_override=true` enables patches mode for spatial Conv2d
    /// even in matrix conv_mode — used by the CROWN-IBP collector (#3813).
    ///
    /// `chunk_override` (#cgan-bn11-chunk): explicit objective row-chunk size
    /// that takes precedence over the `NY_CROWN_OBJ_CHUNK` env knob. `None`
    /// preserves the env-driven behavior byte-for-byte (single pass when the
    /// env is unset/0). The CROWN-IBP collector passes `Some(C)` for targets
    /// whose dense identity pair exceeds the CPU dense budget, so they stream
    /// through the bound-equivalent chunked backward instead of degrading to
    /// IBP.
    ///
    /// `cut_ctx` (#crown-cut-segment): optional backward-to-nearest-bounded-cut
    /// context for the CROWN-IBP sweep. `None` (every non-sweep caller and the
    /// default-OFF gate) keeps the backward walk byte-identical; see
    /// [`CROWN_CUT_SEGMENT_ENV`].
    ///
    /// `expected_fixed_waves` is populated only by the armed M1 scheduler. The
    /// chunk driver verifies that its live route still equals this retained
    /// plan before executing; `None` preserves every historical caller.
    #[allow(clippy::too_many_arguments)] // Backward CROWN dispatch requires all parameters; bundling into a struct would obscure the per-call-site engine threading that #3549 fixes.
    pub(in crate::network::graph_alpha) fn propagate_crown_to_node_core(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        collector_patches_override: bool,
        chunk_override: Option<usize>,
        cut_ctx: Option<&CrownCutContext>,
        expected_fixed_waves: Option<ObjectiveChunkFixedWavePlan>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_to_node_core_impl(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            engine,
            label,
            per_node_deadline,
            per_node_deadline.is_some(),
            collector_patches_override,
            chunk_override,
            cut_ctx,
            false,
            expected_fixed_waves,
        )?
        .into_strict_result()
    }

    /// Collector-only variant that can retain fully completed objective chunks
    /// when a deadline truncates a chunked target walk.
    ///
    /// The distinct return type is the authority boundary: no ordinary
    /// CROWN/alpha-CROWN caller can accidentally consume a partial target as a
    /// complete CROWN result.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::network::graph_alpha) fn propagate_crown_to_node_core_for_collector(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        deadline_is_hard: bool,
        collector_patches_override: bool,
        chunk_override: Option<usize>,
        cut_ctx: Option<&CrownCutContext>,
        deadline_salvage_policy: PartialCrownDeadlineSalvagePolicy,
        expected_fixed_waves: Option<ObjectiveChunkFixedWavePlan>,
    ) -> Result<TargetCrownCollectionResult> {
        self.propagate_crown_to_node_core_impl(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            None,
            engine,
            label,
            per_node_deadline,
            deadline_is_hard,
            collector_patches_override,
            chunk_override,
            cut_ctx,
            deadline_salvage_policy.is_enabled(),
            expected_fixed_waves,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn propagate_crown_to_node_core_impl(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        deadline_is_hard: bool,
        collector_patches_override: bool,
        chunk_override: Option<usize>,
        cut_ctx: Option<&CrownCutContext>,
        salvage_deadline_chunks: bool,
        expected_fixed_waves: Option<ObjectiveChunkFixedWavePlan>,
    ) -> Result<TargetCrownCollectionResult> {
        let relevant_nodes = self.ancestors(target_node)?;

        if relevant_nodes.is_empty() {
            if expected_fixed_waves.is_some() {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{label}: M1 admitted a fixed-wave route for target '{target_node}', \
                     but the target has no backward path"
                )));
            }
            return Ok(TargetCrownCollectionResult::Complete(input.clone()));
        }

        // A target-refresh deadline is authoritative. Backends such as the
        // current CUDA replay path cannot be interrupted once launched, so
        // refuse this optional CROWN refresh and let callers retain their
        // sound IBP/reference bounds instead of overrunning the verifier.
        if deadline_is_hard
            && !crate::sound_gpu_gate::gpu_crown_route_honors_deadline(engine, per_node_deadline)
        {
            return Err(NyError::SoundnessRefusal(
                "deadline-scored target backward requires a cooperative GPU backend".to_string(),
            ));
        }

        // #w4-refresh-deadline: scope the cooperative GPU deadline over this whole
        // per-target backward (the GPU suffix and resnet dispatches below run wide
        // spec-batched / deep resident walks that can only stop BETWEEN batches or
        // layers). Set on the routed backend, always cleared on scope exit; an
        // expired check surfaces as DeadlineExceeded, which every caller handles
        // with a sound reference/IBP fallback.
        let _gpu_deadline_scope =
            crate::sound_gpu_gate::GpuCrownDeadlineScope::set(engine, per_node_deadline);

        let target_bounds = ibp_bounds.get(target_node).ok_or_else(|| {
            NyError::InvalidSpec(format!("Target node {} not in IBP bounds", target_node))
        })?;
        let target_contract = GraphTargetShapeContract::from_bounds(target_node, target_bounds);
        let target_dim = target_contract.flat_dim();
        let input_dim = input.len();

        // PRIMARY (#patches-obj-chunk): objective-chunking streaming, gated OFF
        // by default. When the effective chunk size C > 0 and the seed spans
        // more than one chunk, stream the objective rows in C-row slices and
        // reuse the (objective-independent) GpuSuffixPlan across chunks. When
        // C == 0 we fall through to the single-pass path, which is byte-for-byte
        // the prior behavior. Both paths share the same backward-loop body via
        // `run_target_backward_pass`, so the per-row math is identical.
        //
        // The effective chunk size is the explicit `chunk_override` when given
        // (#cgan-bn11-chunk: the CROWN-IBP collector's memory-budget reroute),
        // otherwise the `NY_CROWN_OBJ_CHUNK` env knob. The chunk decision runs
        // BEFORE seed construction: chunking exists precisely so an over-budget
        // target never materializes its full `[dim x dim]` dense identity pair
        // (6.6 GB for cgan_2023's 28,800-dim BatchNormalization_11).
        let requested_chunk_size = chunk_override.unwrap_or_else(crown_obj_chunk_size);
        let has_conv_transpose = relevant_nodes.iter().any(|name| {
            self.nodes.get(name).is_some_and(|node| {
                matches!(
                    node.layer,
                    Layer::ConvTranspose1d(_) | Layer::ConvTranspose2d(_)
                )
            })
        });
        let chunk_route = objective_chunk_route_plan(
            requested_chunk_size,
            per_node_deadline.is_some(),
            has_conv_transpose,
        );
        let chunk_size = chunk_route.effective_initial_rows;
        if chunk_size > 0 && target_dim > chunk_size {
            let allow_patches = target_allows_patches_start(
                self,
                target_node,
                alpha_state,
                &relevant_nodes,
                target_bounds,
                collector_patches_override,
            );
            let gpu_suffix_plan = GpuSuffixPlan::build(
                &relevant_nodes,
                self,
                input,
                crown_bounds,
                ibp_bounds,
                alpha_state,
            );
            return self.propagate_crown_to_node_chunked(
                input,
                target_node,
                crown_bounds,
                ibp_bounds,
                alpha_state,
                engine,
                label,
                per_node_deadline,
                deadline_is_hard,
                collector_patches_override,
                &relevant_nodes,
                target_bounds,
                &target_contract,
                target_dim,
                input_dim,
                allow_patches,
                &gpu_suffix_plan,
                chunk_size,
                // #chunk-grow ceiling. A caller/env chunk request is a MEMORY
                // bound (the collector's `#cgan-bn11-chunk` auto reroute sizes
                // it from `auto_objective_chunk_rows`, and `NY_CROWN_OBJ_CHUNK`
                // is an explicit request), so widening must never exceed it.
                // When the request is 0 the ONLY reason this walk is chunked is
                // the deadline-granularity cap, and the single-pass width was
                // already memory-admissible — the full objective is the ceiling.
                if chunk_route.requested_rows > 0 {
                    chunk_route.requested_rows.max(chunk_size)
                } else {
                    target_dim
                },
                cut_ctx,
                salvage_deadline_chunks,
                expected_fixed_waves,
            );
        }

        if let Some(expected) = expected_fixed_waves {
            return Err(NyError::UnsupportedConfiguration(format!(
                "{label}: M1 fixed-wave plan {expected:?} for target '{target_node}' \
                 did not reach the objective-chunk driver"
            )));
        }

        let (allow_patches, initial_bounds) = initial_target_crown_bounds_with_override(
            self,
            target_node,
            alpha_state,
            &relevant_nodes,
            target_bounds,
            &target_contract,
            collector_patches_override,
            per_node_deadline,
        )?;
        let gpu_suffix_plan = // O(N) plan replaces O(N*K) rescans (#4340)
            GpuSuffixPlan::build(&relevant_nodes, self, input, crown_bounds, ibp_bounds, alpha_state);

        // Single-pass: seed the full objective and run one backward pass. The
        // `None` result means no input contribution accumulated, in which case
        // the target bounds pass through unchanged (byte-for-byte the prior
        // `target_bounds.clone()` terminal branch).
        let produced = self.run_target_backward_pass(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            engine,
            label,
            per_node_deadline,
            deadline_is_hard,
            collector_patches_override,
            relevant_nodes.as_slice(),
            &target_contract,
            target_dim,
            input_dim,
            allow_patches,
            &gpu_suffix_plan,
            initial_bounds,
            cut_ctx,
        )?;
        if per_node_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
            return Err(NyError::DeadlineExceeded(format!(
                "{label}: per-node deadline exceeded after backward pass for target '{target_node}'"
            )));
        }
        match produced {
            Some(bounds) => Ok(TargetCrownCollectionResult::Complete(bounds)),
            None => Ok(TargetCrownCollectionResult::Complete(target_bounds.clone())),
        }
    }

    /// Opt-in target-to-input CROWN certificate.
    ///
    /// Returns identity-seeded target rows in flat target-output order as
    /// `lower_a * input + lower_b <= target <= upper_a * input + upper_b`.
    /// Any certified coefficient error produced by the backward pass is folded
    /// outward into the biases over the exact supplied `input` box before this
    /// public method returns; a residual error channel is rejected fail-closed.
    /// Consequently the returned raw coefficients are valid only on that exact
    /// box and its subsets. They must not be reused on a wider or unrelated box.
    ///
    /// This path is deliberately separate from `propagate_crown_to_node`: the
    /// existing concrete API keeps its bounds-only GPU suffixes and concrete
    /// finalization unchanged. Linear capture runs the same CPU/Patches backward
    /// steps in deterministic identity-row chunks, with a checked hard cap on
    /// the aggregate full assembly plus one conservative chunk payload.
    #[allow(clippy::too_many_arguments)]
    pub fn propagate_crown_input_linear_to_node(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<LinearBounds> {
        self.propagate_crown_input_linear_to_node_with_limits(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            engine,
            deadline,
            TargetInputLinearLimits::PRODUCTION,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn propagate_crown_input_linear_to_node_with_limits(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<std::time::Instant>,
        limits: TargetInputLinearLimits,
    ) -> Result<LinearBounds> {
        let mut linear = self.capture_crown_input_linear_to_node_raw_with_limits(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            engine,
            deadline,
            limits,
        )?;
        let flat_input = input.flatten();
        let input_lower = flat_input.lower().as_slice().ok_or_else(|| {
            NyError::InternalError(
                "target input-linear CROWN flattened lower input is not contiguous".to_string(),
            )
        })?;
        let input_upper = flat_input.upper().as_slice().ok_or_else(|| {
            NyError::InternalError(
                "target input-linear CROWN flattened upper input is not contiguous".to_string(),
            )
        })?;
        linear.fold_coeff_err_into_bias(input_lower, input_upper);
        if linear.has_coeff_err() {
            return Err(NyError::InternalError(format!(
                "target input-linear CROWN retained coefficient error after exact-box fold \
                 for target '{target_node}'"
            )));
        }
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "target input-linear CROWN deadline exceeded while folding target \
                 '{target_node}'"
            )));
        }
        Ok(linear)
    }

    /// Private raw capture. The coefficient-error channel is intentionally
    /// inaccessible outside this module's implementation/tests; the public API
    /// above discharges it over the exact supplied input box.
    #[allow(clippy::too_many_arguments)]
    fn capture_crown_input_linear_to_node_raw_with_limits(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<std::time::Instant>,
        limits: TargetInputLinearLimits,
    ) -> Result<LinearBounds> {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "target input-linear CROWN deadline expired before target '{target_node}'"
            )));
        }
        // This opt-in certificate path assembles six full target-by-input
        // arrays, scans/clones target ancestry, and performs whole-array chunk
        // copies and error folding. Those legacy phases cannot yet observe a
        // finite authority. Decline before graph lookup, dimension arithmetic,
        // or allocation; callers may retain their existing certified fallback.
        // The no-deadline certificate path remains byte-for-byte unchanged.
        if deadline.is_some() {
            return Err(NyError::UnsupportedConfiguration(
                "finite target input-linear CROWN capture is not cooperatively bounded".to_string(),
            ));
        }
        if target_node != NETWORK_INPUT && !self.nodes.contains_key(target_node) {
            return Err(NyError::InvalidSpec(format!(
                "target input-linear CROWN target '{target_node}' is not a graph node"
            )));
        }

        let input_dim = input.len();
        let target_dim = if target_node == NETWORK_INPUT {
            input_dim
        } else {
            ibp_bounds
                .get(target_node)
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!("Target node {target_node} not in IBP bounds"))
                })?
                .len()
        };
        let fixed_bytes =
            target_input_linear_fixed_bytes(target_dim, input_dim).unwrap_or(usize::MAX);
        if fixed_bytes > limits.aggregate_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: fixed_bytes,
                budget_bytes: limits.aggregate_bytes,
                site: "graph-alpha target input-linear aggregate workspace",
            });
        }
        if target_node == NETWORK_INPUT {
            return Ok(LinearBounds::identity(input_dim));
        }

        let target_bounds = ibp_bounds.get(target_node).ok_or_else(|| {
            NyError::InvalidSpec(format!("Target node {target_node} not in IBP bounds"))
        })?;
        let relevant_nodes = self.ancestors(target_node)?;
        if relevant_nodes.is_empty() {
            return Err(NyError::InvalidSpec(format!(
                "target input-linear CROWN graph node '{target_node}' has no backward ancestry"
            )));
        }
        let mut relation_cols_sum = target_dim.saturating_add(input_dim);
        let mut relation_count = 2usize;
        for name in &relevant_nodes {
            let width = crown_bounds
                .get(name)
                .or_else(|| ibp_bounds.get(name))
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "target input-linear CROWN relation node '{name}' has no bound box"
                    ))
                })?
                .len();
            relation_cols_sum = relation_cols_sum.saturating_add(width);
            relation_count = relation_count.saturating_add(1);
        }
        let chunk_row_bytes =
            target_input_linear_chunk_row_bytes(target_dim, relation_cols_sum, relation_count)
                .unwrap_or(usize::MAX);
        let has_conv_transpose = relevant_nodes.iter().any(|name| {
            self.nodes.get(name).is_some_and(|node| {
                matches!(
                    node.layer,
                    Layer::ConvTranspose1d(_) | Layer::ConvTranspose2d(_)
                )
            })
        });
        let chunk_rows = target_input_linear_chunk_rows(
            target_dim,
            fixed_bytes,
            chunk_row_bytes,
            deadline.is_some(),
            has_conv_transpose,
            limits.aggregate_bytes,
        )?;

        let _gpu_deadline_scope =
            crate::sound_gpu_gate::GpuCrownDeadlineScope::set(engine, deadline);
        let allow_patches = target_allows_patches_start(
            self,
            target_node,
            None,
            &relevant_nodes,
            target_bounds,
            false,
        );
        let target_shape = target_bounds.shape();
        let spatial = if allow_patches && target_shape.len() == 3 {
            Some((target_shape[0], target_shape[1], target_shape[2]))
        } else {
            None
        };
        let gpu_suffix_plan =
            GpuSuffixPlan::build(&relevant_nodes, self, input, crown_bounds, ibp_bounds, None);

        // Allocate the complete, preflighted return payload once. Chunks are
        // copied directly into their deterministic row ranges, so we never hold
        // a second full certificate during concatenation.
        let mut lower_a = Array2::<f32>::zeros((target_dim, input_dim));
        let mut upper_a = Array2::<f32>::zeros((target_dim, input_dim));
        let mut lower_b = Array1::<f32>::zeros(target_dim);
        let mut upper_b = Array1::<f32>::zeros(target_dim);
        let mut lower_err = Array2::<f32>::zeros((target_dim, input_dim));
        let mut upper_err = Array2::<f32>::zeros((target_dim, input_dim));
        let mut any_err = false;

        let mut r0 = 0usize;
        while r0 < target_dim {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Err(NyError::DeadlineExceeded(format!(
                    "target input-linear CROWN deadline exceeded before rows {r0}.. for \
                     target '{target_node}'"
                )));
            }
            let r1 = (r0 + chunk_rows).min(target_dim);
            let rows = r1 - r0;
            let chunk_bytes = chunk_row_bytes
                .checked_mul(rows)
                .and_then(|bytes| fixed_bytes.checked_add(bytes))
                .unwrap_or(usize::MAX);
            if chunk_bytes > limits.aggregate_bytes {
                return Err(NyError::CpuMemoryExceeded {
                    required_bytes: chunk_bytes,
                    budget_bytes: limits.aggregate_bytes,
                    site: "graph-alpha target input-linear aggregate workspace",
                });
            }
            let seed = build_chunk_seed(target_dim, r0, r1, spatial, deadline)?;
            let chunk_contract =
                GraphTargetShapeContract::from_bounds(target_node, &flat_bounds_view(rows)?);
            let produced = self.run_target_backward_pass_linear(
                input,
                target_node,
                crown_bounds,
                ibp_bounds,
                None,
                engine,
                "CROWN target input-linear",
                deadline,
                false,
                relevant_nodes.as_slice(),
                &chunk_contract,
                rows,
                input_dim,
                allow_patches,
                &gpu_suffix_plan,
                seed,
                None,
            )?;
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Err(NyError::DeadlineExceeded(format!(
                    "target input-linear CROWN deadline exceeded after rows {r0}..{r1} \
                     for target '{target_node}'"
                )));
            }

            let chunk = match produced {
                Some(linear) => linear,
                None => {
                    // Mirror the concrete API's no-input-contribution fallback:
                    // the certified target box is a zero-coefficient affine
                    // enclosure over the original input.
                    let lo = Array1::from_iter(
                        target_bounds.lower().iter().copied().skip(r0).take(rows),
                    );
                    let up = Array1::from_iter(
                        target_bounds.upper().iter().copied().skip(r0).take(rows),
                    );
                    LinearBounds::new(
                        Array2::zeros((rows, input_dim)),
                        lo,
                        Array2::zeros((rows, input_dim)),
                        up,
                    )?
                }
            };
            if chunk.num_outputs() != rows || chunk.num_inputs() != input_dim {
                return Err(NyError::InvalidSpec(format!(
                    "target input-linear CROWN rows {r0}..{r1} produced shape {}x{}, \
                     expected {rows}x{input_dim}",
                    chunk.num_outputs(),
                    chunk.num_inputs()
                )));
            }
            lower_a
                .slice_mut(ndarray::s![r0..r1, ..])
                .assign(chunk.lower_a());
            upper_a
                .slice_mut(ndarray::s![r0..r1, ..])
                .assign(chunk.upper_a());
            lower_b
                .slice_mut(ndarray::s![r0..r1])
                .assign(chunk.lower_b());
            upper_b
                .slice_mut(ndarray::s![r0..r1])
                .assign(chunk.upper_b());
            if let Some(err) = chunk.lower_a_err() {
                if err.iter().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err(NyError::NumericalInstability(format!(
                        "target input-linear CROWN lower coefficient error is invalid in \
                         rows {r0}..{r1}"
                    )));
                }
                any_err = true;
                lower_err.slice_mut(ndarray::s![r0..r1, ..]).assign(err);
            }
            if let Some(err) = chunk.upper_a_err() {
                if err.iter().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err(NyError::NumericalInstability(format!(
                        "target input-linear CROWN upper coefficient error is invalid in \
                         rows {r0}..{r1}"
                    )));
                }
                any_err = true;
                upper_err.slice_mut(ndarray::s![r0..r1, ..]).assign(err);
            }
            r0 = r1;
        }

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "target input-linear CROWN deadline exceeded while assembling target \
                 '{target_node}'"
            )));
        }
        // Strict construction fails closed without allocating a second
        // conservative A/b payload while the four assembled A/error matrices
        // are still resident under the aggregate cap.
        let mut assembled = LinearBounds::new(lower_a, lower_b, upper_a, upper_b)?;
        if any_err {
            assembled.set_coeff_err(lower_err, upper_err);
        }
        Ok(assembled)
    }

    /// Run one backward CROWN pass for a single seed (`initial_bounds`) over
    /// `relevant_nodes`, returning the concretized + shape-restored bounds for
    /// the seed's rows, or `None` when no input contribution accumulated.
    ///
    /// This is the verbatim per-target backward loop. Both the single-pass and
    /// the objective-chunking (#patches-obj-chunk) paths drive it; chunking only
    /// changes the seed (a C-row slice of the objective) and the restore shape
    /// (a flat `[chunk_rows]` contract), never the per-step relaxation math.
    #[allow(clippy::too_many_arguments)]
    fn run_target_backward_pass(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        deadline_is_hard: bool,
        collector_patches_override: bool,
        relevant_nodes: &[String],
        // Restore contract: full target shape for single-pass, flat `[chunk_rows]`
        // for objective-chunking. Used by both the GPU-suffix and final concretize.
        target_contract: &GraphTargetShapeContract,
        // Number of objective rows in THIS seed (full target_dim, or chunk size).
        target_dim: usize,
        input_dim: usize,
        allow_patches: bool,
        gpu_suffix_plan: &GpuSuffixPlan,
        initial_bounds: CrownBounds,
        cut_ctx: Option<&CrownCutContext>,
    ) -> Result<Option<BoundedTensor>> {
        match self.run_target_backward_pass_core(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            engine,
            label,
            per_node_deadline,
            deadline_is_hard,
            collector_patches_override,
            relevant_nodes,
            target_contract,
            target_dim,
            input_dim,
            allow_patches,
            gpu_suffix_plan,
            initial_bounds,
            cut_ctx,
            None,
            false,
        )? {
            TargetBackwardPassResult::NoInputContribution => Ok(None),
            TargetBackwardPassResult::Concrete(bounds) => Ok(Some(bounds)),
            TargetBackwardPassResult::InputLinear(_) => Err(NyError::InternalError(
                "concrete target backward unexpectedly returned an input-linear certificate"
                    .to_string(),
            )),
        }
    }

    /// Linear-capture sibling of [`run_target_backward_pass`]. It deliberately
    /// skips bounds-only GPU suffixes and returns the final relation at
    /// `NETWORK_INPUT` with certified coefficient errors still attached.
    #[allow(clippy::too_many_arguments)]
    fn run_target_backward_pass_linear(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        collector_patches_override: bool,
        relevant_nodes: &[String],
        target_contract: &GraphTargetShapeContract,
        target_dim: usize,
        input_dim: usize,
        allow_patches: bool,
        gpu_suffix_plan: &GpuSuffixPlan,
        initial_bounds: CrownBounds,
        cut_ctx: Option<&CrownCutContext>,
    ) -> Result<Option<LinearBounds>> {
        match self.run_target_backward_pass_core(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            engine,
            label,
            per_node_deadline,
            per_node_deadline.is_some(),
            collector_patches_override,
            relevant_nodes,
            target_contract,
            target_dim,
            input_dim,
            allow_patches,
            gpu_suffix_plan,
            initial_bounds,
            cut_ctx,
            None,
            true,
        )? {
            TargetBackwardPassResult::NoInputContribution => Ok(None),
            TargetBackwardPassResult::InputLinear(linear) => Ok(Some(linear)),
            TargetBackwardPassResult::Concrete(_) => Err(NyError::InternalError(
                "input-linear target backward unexpectedly returned concrete bounds".to_string(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_target_backward_pass_core(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        deadline_is_hard: bool,
        collector_patches_override: bool,
        relevant_nodes: &[String],
        target_contract: &GraphTargetShapeContract,
        target_dim: usize,
        input_dim: usize,
        allow_patches: bool,
        gpu_suffix_plan: &GpuSuffixPlan,
        initial_bounds: CrownBounds,
        cut_ctx: Option<&CrownCutContext>,
        additional_seed: Option<AdditionalTargetBackwardSeed<'_>>,
        capture_input_linear: bool,
    ) -> Result<TargetBackwardPassResult> {
        let has_additional_seed = additional_seed.is_some();
        if has_additional_seed {
            if allow_patches {
                return Err(NyError::SoundnessRefusal(
                    "output-conditioned additional-seed backward is dense-only; patches admission \
                     is forbidden"
                        .to_string(),
                ));
            }
            for node_name in relevant_nodes {
                let node = self.nodes.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "output-conditioned additional-seed ancestry references missing node \
                         '{node_name}'"
                    ))
                })?;
                let relevant_alpha_relu = matches!(&node.layer, Layer::ReLU(_))
                    && alpha_state
                        .and_then(|state| state.alpha(node_name))
                        .is_some();
                if relevant_alpha_relu
                    || !output_conditioned_additional_seed_node_is_audited(
                        &node.layer,
                        node.inputs.len(),
                    )
                {
                    let reason = output_conditioned_additional_seed_refusal_reason(
                        &node.layer,
                        node_name,
                        alpha_state,
                    );
                    return Err(NyError::SoundnessRefusal(format!(
                        "output-conditioned additional-seed backward refused node '{node_name}' \
                         ({}): {reason}",
                        node.layer.layer_type()
                    )));
                }
            }
        }
        // Resnet-aware GPU-resident suffix (#vnncomp-resnet): if the whole ancestor
        // suffix decomposes into clean chains + identity/projection residual blocks,
        // run it on the proven sound GPU-resident resnet backward in one shot —
        // avoiding the CPU dense fork that materializes `[num_objectives × conv_dim]`
        // host matrices and OOMs/times out on cifar100/tinyimagenet ResNets. Every
        // bail (non-decomposable suffix, no sound GPU engine, GPU error, NaN) falls
        // through to the proven-sound CPU dense backward below, so the 0-wrong moat
        // holds. Only attempted from a fresh frontier (the loop has not run yet).
        //
        // The cheap `target_dim` (objective count) gate runs FIRST: a full dense GPU
        // resnet backward is only worthwhile for the verdict-shaped backward (few
        // objectives). Crucially this also avoids densifying a wide intermediate
        // node's identity seed (a `[dim × dim]` blow-up) — the gate must precede the
        // `into_dense()` below.
        if !capture_input_linear
            && !has_additional_seed
            && super::super::resnet_decompose::resnet_gpu_enabled()
            && target_dim <= super::super::resnet_decompose::resnet_gpu_max_objectives()
        {
            let seed_lb = match materialize_optional_resnet_seed(&initial_bounds, per_node_deadline)
            {
                Ok(seed) => Some(seed),
                Err(error) if error.is_deadline_exceeded() || error.is_cpu_memory_exceeded() => {
                    // Resource failures are not optional-acceleration misses:
                    // preserve the absolute deadline and let the caller's
                    // established target-level memory fallback select IBP.
                    return Err(error);
                }
                Err(error) => {
                    debug!(
                        "{label}: optional GPU-resnet seed materialization failed for target \
                         '{target_node}' ({error}); continuing on the CPU backward route"
                    );
                    None
                }
            };
            if let Some(seed_lb) = seed_lb {
                if let Some(bounds) = super::super::resnet_decompose::try_resnet_gpu_suffix(
                    self,
                    input,
                    target_node,
                    crown_bounds,
                    ibp_bounds,
                    alpha_state,
                    engine,
                    per_node_deadline,
                    seed_lb.as_ref(),
                )? {
                    return Ok(TargetBackwardPassResult::Concrete(
                        target_contract.restore_concrete(
                            bounds,
                            "Graph alpha-CROWN GPU resnet suffix restore",
                        )?,
                    ));
                }
            }
        }

        let mut node_crown_bounds = CrownMergeAccumulator::new();
        node_crown_bounds.insert(target_node.to_string(), initial_bounds);
        if let Some(seed) = additional_seed {
            if seed.node_name == target_node || node_crown_bounds.contains_key(seed.node_name) {
                return Err(NyError::InvalidSpec(
                    "output-conditioned two-seed backward requires distinct seed nodes".to_string(),
                ));
            }
            node_crown_bounds.insert(seed.node_name.to_string(), seed.bounds);
        }
        let mut input_accumulated = false;
        // #crown-nonfinite-tripwire (NY_CROWN_GAIN=1): report the FIRST backward
        // step at which the accumulated relation goes non-finite.
        //
        // A target whose bias reaches +/-inf concretizes to [-inf, inf], which the
        // IBP intersection then silently replaces with the IBP box while the
        // provenance still records `Crown` — so the node looks tightened and
        // contributes nothing. Measured on yolo_2023: Conv_12, Add_15, Conv_20 and
        // Conv_25 all land there while Conv_5 and Add_8 complete the same kind of
        // walk finitely (15x / 14.7x gains). The concretize-time guard only tests
        // the INCOMING bias, so it reports the infinity rather than creating it;
        // this says which step actually produced it.
        let mut nonfinite_reported = false;
        let mut prev_step: Option<String> = None;
        // #pass-cost (NY_PHASE_TELEMETRY=1): price EVERY backward step of ONE
        // target pass. Measured evidence says this pass's cost is ~97%
        // independent of the objective-row count k (rows fell 18.7x for 1.57x
        // wall), so the dominant term must scale with something else — this
        // probe says what, per layer: carrier kind, logical rows, coefficient
        // width, and wall time.
        let pass_probe = crate::phase_telemetry::phase_telemetry_enabled();
        // #patches-drop (dark, NY_PATCHES_CARRIER_TRACE=1, print-only): publish
        // this walk's position so a `[patches-drop]` line emitted deep inside
        // the materializer can be attributed to (target, node).
        let carrier_trace = crate::patches_carrier_trace::enabled();
        let pass_start = std::time::Instant::now();
        let mut step_costs: Vec<(String, String, usize, usize, f64)> = Vec::new();
        for node_name in relevant_nodes.iter().rev() {
            let step_start = std::time::Instant::now();
            if let Some(deadline) = per_node_deadline {
                if std::time::Instant::now() >= deadline {
                    return Err(NyError::DeadlineExceeded(format!(
                        "{}: per-node deadline exceeded at backward step '{}' for target '{}'",
                        label, node_name, target_node
                    )));
                }
            }

            let node = match self.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            let mut node_cb =
                match node_crown_bounds.take_with_deadline(node_name, per_node_deadline)? {
                    Some(cb) => cb,
                    None => continue,
                };
            if carrier_trace {
                crate::patches_carrier_trace::enter_node(target_node, node_name);
            }
            // #pass-cost: snapshot the carrier shape BEFORE the step consumes it.
            let step_shape: (&'static str, usize, usize) = if pass_probe {
                match &node_cb {
                    CrownBounds::Dense(lb) => ("dense", lb.num_outputs(), lb.num_inputs()),
                    CrownBounds::Patches(pb) => {
                        let w = pb
                            .lower_a
                            .patches
                            .as_ref()
                            .map_or(0, |p| p.len() / pb.row_count.max(1));
                        ("patches", pb.row_count, w)
                    }
                }
            } else {
                ("", 0, 0)
            };

            if crown_gain_probe_enabled() && !nonfinite_reported {
                if let Some(kind) = crown_bounds_nonfinite_kind(&node_cb) {
                    warn!(
                        "#crown-nonfinite target='{}' detected_at='{}' produced_by='{}' component={} (this target will concretize to [-inf, inf])",
                        target_node,
                        node_name,
                        prev_step.as_deref().unwrap_or("<target-seed>"),
                        kind
                    );
                    nonfinite_reported = true;
                }
            }
            prev_step = Some(node_name.clone());
            if tracing::enabled!(tracing::Level::TRACE) {
                if let CrownBounds::Dense(ref lb) = node_cb {
                    let bad = lb
                        .lower_a()
                        .iter()
                        .chain(lb.upper_a().iter())
                        .filter(|v| !v.is_finite())
                        .count();
                    let bad_b = lb
                        .lower_b()
                        .iter()
                        .chain(lb.upper_b().iter())
                        .filter(|v| !v.is_finite())
                        .count();
                    tracing::trace!(
                        "target-backward step '{}' ({}) seed dense a_bad={} b_bad={} rows={} cols={}",
                        node_name,
                        node.layer.layer_type(),
                        bad,
                        bad_b,
                        lb.num_outputs(),
                        lb.num_inputs()
                    );
                } else {
                    tracing::trace!(
                        "target-backward step '{}' ({}) seed patches",
                        node_name,
                        node.layer.layer_type()
                    );
                }
            }

            // #crown-cut-segment (NY_CROWN_CUT_SEGMENT): backward-to-nearest-
            // bounded-cut. When the walk reaches a designated cut node — an
            // EARLIER node whose bounds this sweep already finalized — the
            // accumulated linear relation is concretized against that node's
            // bound-box with the SAME directed-rounding concretization the
            // input-box path uses (`concretize_sound`, which also folds any
            // carried certified coefficient error over the box), and this path
            // of the walk stops instead of expanding the node's prefix.
            //
            // SOUNDNESS: the swept box is a valid enclosure of the cut node's
            // reachable set, so the concretized interval encloses the
            // relation's value for every reachable input — this is exactly the
            // input-box concretization applied at an intermediate cut (the
            // input box is the trivial cut). The result is only ever LOOSER
            // than full-prefix expansion (the box drops inter-node
            // correlations). Per-node relaxations of expanded nodes are
            // untouched — only the DEPTH of the walk changes.
            //
            // FAIL-OPEN: a missing, non-finite, or shape-mismatched box means
            // the node is expanded exactly as with the gate off (slower exact
            // behavior, never a wrong bound).
            // Cut selection starts with a full endpoint scan of the retained
            // box. Until that scan is cooperative, a finite target walk keeps
            // the ordinary prefix expansion and never enters the cut seam.
            if let Some(ctx) = cut_ctx.filter(|_| !deadline_is_hard) {
                if node_name != target_node && ctx.is_cut(node_name) {
                    let cut_box = crown_bounds.get(node_name).filter(|b| box_is_finite(b));
                    // A Patches-carried relation must densify to concretize
                    // here; only cut when its dense pair fits the CPU dense
                    // budget (a Dense relation already paid that memory, so it
                    // always cuts). Over-budget patches relations keep the
                    // gate-off patches walk (fail-open: exact, no new
                    // allocation cliff).
                    let densify_fits = matches!(node_cb, CrownBounds::Dense(_))
                        || cut_box.is_some_and(|b| {
                            crate::network::crown_memory::dense_pair_bytes(target_dim, b.len())
                                .is_some_and(|bytes| bytes <= cpu_crown_dense_budget_bytes())
                        });
                    if let (Some(cut_box), true) = (cut_box, densify_fits) {
                        let node_lb = node_cb.into_dense_with_deadline(per_node_deadline)?;
                        if node_lb.num_inputs() == cut_box.len() {
                            let concrete = node_lb
                                .concretize_sound_with_deadline(cut_box, per_node_deadline)?;
                            let (lower, upper) = concrete_into_ix1(
                                concrete,
                                "graph-alpha crown cut-segment concretize",
                            )?;
                            Self::accumulate_bias_to_network_input_crown_with_deadline(
                                &lower,
                                &upper,
                                &mut node_crown_bounds,
                                target_dim,
                                input_dim,
                                &mut input_accumulated,
                                per_node_deadline,
                            )?;
                            ctx.record_cut();
                            continue;
                        }
                        // Defensive shape drift: keep the (already densified)
                        // relation and expand through the node as with the
                        // gate off.
                        node_cb = CrownBounds::Dense(node_lb);
                    }
                }
            }

            // Resolve first input; multi-input nodes skip to dedicated branches (#4112).
            let (first_input_name, pre_activation) = if node.inputs.len() == 1 {
                let name = node.require_unary_input().map_err(|_| {
                    NyError::InvalidSpec(format!(
                        "{label} failed at '{node_name}' ({}): node has no inputs",
                        node.layer.layer_type()
                    ))
                })?;
                (
                    name,
                    resolve_preactivation(input, name, crown_bounds, ibp_bounds)?,
                )
            } else {
                let fallback = node
                    .inputs
                    .first()
                    .map(String::as_str)
                    .unwrap_or(NETWORK_INPUT);
                (fallback, input)
            };

            // #conv-crown-residual: a 2-input elementwise Add/Sub is consumed by
            // the patches-native residual passthrough (relation duplicated down
            // both branches, summed at the join). It must NOT go through
            // `try_patches_target_step_core`, whose generic tail accumulates the
            // whole relation to `first_input_name` only.
            //
            // Gating this on `inputs.len() == 1` was the deep-conv wall: every
            // ResNet backward walk densified at its first residual Add regardless
            // of receptive-field size, and IBP width then compounds
            // multiplicatively through the remaining convs
            // (docs/PATCHES_RESIDUAL_ADD_ROOT_CAUSE_2026-07-27.md).
            let was_patches = matches!(&node_cb, CrownBounds::Patches(_));
            let patches_step_handled = if !allow_patches {
                false
            } else if node.inputs.len() == 1 {
                try_patches_target_step_core(
                    self,
                    label,
                    node_name,
                    node,
                    &mut node_cb,
                    first_input_name,
                    pre_activation,
                    ibp_bounds,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                    engine,
                    per_node_deadline,
                    deadline_is_hard,
                    collector_patches_override,
                    alpha_state,
                )?
            } else {
                try_patches_residual_target_step(
                    self,
                    label,
                    node,
                    &mut node_cb,
                    ibp_bounds,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                    per_node_deadline,
                    deadline_is_hard,
                )?
            };
            // #patches-drop: the decision tuple this regression hunt kept
            // reconstructing by hand. `repr_in=patches hard=1 allow_patches=1
            // handled=0` is the shape that goes on to densify. `was_patches`
            // prints as `repr_in=` because it was sampled above, BEFORE
            // `try_patches_target_step_core` ran — it is the step's incoming
            // representation, not its result.
            if carrier_trace {
                crate::patches_carrier_trace::record_step_decision(
                    target_node,
                    node_name,
                    node.layer.layer_type(),
                    was_patches,
                    allow_patches,
                    deadline_is_hard,
                    patches_step_handled,
                    per_node_deadline,
                    crate::network::core::sequential::crown::patches_step::expiry_authority_armed(),
                );
            }
            if patches_step_handled {
                continue;
            }
            // Preserve carrier provenance separately from the scheduling
            // timestamp. Only a hard finite request may turn an actual
            // Patches->Dense fallthrough into strict proof authority; internal
            // soft deadlines retain the historical one-unit Dense dispatch.
            // Same question as the dispatcher and the patches family: EXPIRY,
            // not presence, when the expiry-authority arm is in force.
            let finite_structured_boundary = was_patches
                && deadline_is_hard
                && (!crate::network::core::sequential::crown::patches_step::expiry_authority_armed(
                ) || per_node_deadline.is_some_and(|limit| std::time::Instant::now() >= limit));

            // Site-specific branches below bypass the canonical dispatcher and
            // contain opaque nonlinear relaxations, eager error folds, Where
            // scans, and bound-map clones. A Dense carrier that actually came
            // from a hard finite Patches transaction must decline before that
            // work. Ordinary finite Dense carriers retain their historical
            // branch; generic Patches fallthroughs take the strict dispatcher.
            if finite_structured_boundary
                && matches!(
                    &node.layer,
                    Layer::ReLU(_)
                        | Layer::Sigmoid(_)
                        | Layer::Tanh(_)
                        | Layer::Sqrt(_)
                        | Layer::Reciprocal(_)
                        | Layer::MulBinary(_)
                        | Layer::Div(_)
                        | Layer::Where(_)
                )
            {
                return Err(NyError::UnsupportedOp(format!(
                    "{label}: finite Patches target boundary reached non-cooperative custom route '{}' ({})",
                    node_name,
                    node.layer.layer_type()
                )));
            }

            // #patches-row-range: a Patches relation the patches-capable step
            // could not consume must densify here. On VGG-scale conv targets
            // that dense pair reaches TB scale (3.2M x 150K rows) and the
            // allocation aborts the process. Refuse over-budget densification
            // with the structured CpuMemoryExceeded that every caller maps to
            // a sound fallback (IBP / another strategy); under-budget
            // relations keep the existing path byte-for-byte.
            if let CrownBounds::Patches(ref pb) = node_cb {
                if let Some(err) = patches_densify_over_budget(pb, cpu_crown_dense_budget_bytes()) {
                    let (rows, cols) = pb.dense_pair_shape().unwrap_or((0, 0));
                    debug!(
                        "{}: backward step '{}' for target '{}' needs a {}x{} dense pair \
                         over the CPU dense budget; degrading ({})",
                        label, node_name, target_node, rows, cols, err
                    );
                    return Err(err);
                }
            }
            let mut node_lb = node_cb.into_dense_with_deadline(per_node_deadline)?;
            if !capture_input_linear && !has_additional_seed {
                if let Some(bounds) = try_finish_target_gpu_suffix_with_pending_input(
                    input,
                    node_name,
                    &node_lb,
                    gpu_suffix_plan,
                    engine,
                    per_node_deadline,
                    target_contract,
                    &mut node_crown_bounds,
                )? {
                    return Ok(TargetBackwardPassResult::Concrete(bounds));
                }
            }
            if has_additional_seed {
                fold_output_conditioned_additional_seed_error_for_cpu(
                    node_name,
                    &mut node_lb,
                    crown_bounds,
                    ibp_bounds,
                )?;
            }
            if let Layer::ReLU(r) = &node.layer {
                let mut new_lb = if let Some(alpha) = alpha_state.and_then(|s| s.alpha(node_name)) {
                    let alpha_upper = alpha_state.and_then(|s| s.alpha_upper(node_name));
                    let alpha_expanded = alpha_state
                        .map(|s| s.expand_alpha(node_name, alpha))
                        .unwrap_or_else(|| alpha.clone());
                    let alpha_upper_expanded = alpha_state
                        .and_then(|s| alpha_upper.map(|upper| s.expand_alpha(node_name, upper)));
                    let (bounds, _grad, _grad_upper) = r.propagate_linear_with_alpha(
                        &node_lb,
                        pre_activation,
                        &alpha_expanded,
                        alpha_upper_expanded.as_ref(),
                    )?;
                    bounds
                } else {
                    r.propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            NyError::InvalidSpec(format!(
                                "{} failed at '{}' (ReLU): {}",
                                label, node_name, e
                            ))
                        })?
                };
                // Eager per-row discharge of the carried coefficient error over
                // the (CROWN-tightened) pre-activation cut — the tightest box the
                // error will ever see; carrying it further pays IBP-scale
                // magnitudes (#cgan-conv-err-compose, see
                // LinearBounds::fold_coeff_err_over_box_eager).
                new_lb.fold_coeff_err_over_box_eager(pre_activation);
                self.accumulate_dense_bounds_to_input_with_deadline(
                    first_input_name,
                    new_lb,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                    per_node_deadline,
                )?;
            } else if let Layer::Sigmoid(sigmoid) = &node.layer {
                let mut new_lb = match alpha_state
                    .and_then(|s| s.monotone_s_shaped_alpha(node_name))
                {
                    Some(alpha) => {
                        match sigmoid.propagate_linear_with_alpha(&node_lb, pre_activation, alpha) {
                            Ok(bounds) => bounds,
                            // #4118: a monotone alpha bundle whose width disagrees with the
                            // pre-activation (e.g. a stale warm-started bundle reused across a
                            // reshape) surfaces as ShapeMismatch. Fixed-slope CROWN is always a
                            // sound relaxation, so retry this node locally instead of failing the
                            // whole backward pass and triggering a graph-wide fallback.
                            Err(NyError::ShapeMismatch { expected, got }) => {
                                warn!(
                                    "{} monotone alpha shape mismatch at '{}' (Sigmoid): expected {:?}, got {:?}; retrying fixed-slope locally",
                                    label, node_name, expected, got
                                );
                                sigmoid
                                    .propagate_linear_with_bounds(&node_lb, pre_activation)
                                    .map_err(|e| {
                                        NyError::InvalidSpec(format!(
                                            "{} failed at '{}' (Sigmoid fixed-slope retry): {}",
                                            label, node_name, e
                                        ))
                                    })?
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    None => sigmoid
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            NyError::InvalidSpec(format!(
                                "{} failed at '{}' (Sigmoid): {}",
                                label, node_name, e
                            ))
                        })?,
                };
                new_lb.fold_coeff_err_over_box_eager(pre_activation); // #cgan-conv-err-compose
                self.accumulate_dense_bounds_to_input_with_deadline(
                    first_input_name,
                    new_lb,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                    per_node_deadline,
                )?;
            } else if let Layer::Tanh(tanh) = &node.layer {
                let mut new_lb = match alpha_state
                    .and_then(|s| s.monotone_s_shaped_alpha(node_name))
                {
                    Some(alpha) => {
                        match tanh.propagate_linear_with_alpha(&node_lb, pre_activation, alpha) {
                            Ok(bounds) => bounds,
                            // #4118: see the Sigmoid branch — a mismatched monotone alpha bundle
                            // falls back to the sound fixed-slope relaxation for this node only.
                            Err(NyError::ShapeMismatch { expected, got }) => {
                                warn!(
                                    "{} monotone alpha shape mismatch at '{}' (Tanh): expected {:?}, got {:?}; retrying fixed-slope locally",
                                    label, node_name, expected, got
                                );
                                tanh.propagate_linear_with_bounds(&node_lb, pre_activation)
                                    .map_err(|e| {
                                        NyError::InvalidSpec(format!(
                                            "{} failed at '{}' (Tanh fixed-slope retry): {}",
                                            label, node_name, e
                                        ))
                                    })?
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    None => tanh
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            NyError::InvalidSpec(format!(
                                "{} failed at '{}' (Tanh): {}",
                                label, node_name, e
                            ))
                        })?,
                };
                new_lb.fold_coeff_err_over_box_eager(pre_activation); // #cgan-conv-err-compose
                self.accumulate_dense_bounds_to_input_with_deadline(
                    first_input_name,
                    new_lb,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                    per_node_deadline,
                )?;
            } else if let Layer::Sqrt(sqrt) = &node.layer {
                let mut new_lb = sqrt_support::backward_sqrt_node(
                    sqrt,
                    alpha_state,
                    node_name,
                    label,
                    &node_lb,
                    pre_activation,
                )?;
                new_lb.fold_coeff_err_over_box_eager(pre_activation); // #cgan-conv-err-compose
                self.accumulate_dense_bounds_to_input_with_deadline(
                    first_input_name,
                    new_lb,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                    per_node_deadline,
                )?;
            } else if let Layer::Reciprocal(reciprocal) = &node.layer {
                let mut new_lb = reciprocal_support::backward_reciprocal_node(
                    reciprocal,
                    alpha_state,
                    node_name,
                    label,
                    &node_lb,
                    pre_activation,
                )?;
                new_lb.fold_coeff_err_over_box_eager(pre_activation); // #cgan-conv-err-compose
                self.accumulate_dense_bounds_to_input_with_deadline(
                    first_input_name,
                    new_lb,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                    per_node_deadline,
                )?;
            } else if let Layer::MulBinary(mul) = &node.layer {
                let (input_a_name, input_b_name) = node.require_binary_inputs()?;
                let input_a_bounds = if input_a_name == NETWORK_INPUT {
                    input
                } else {
                    crown_bounds
                        .get(input_a_name)
                        .or_else(|| ibp_bounds.get(input_a_name))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "MulBinary input A '{}' not found",
                                input_a_name
                            ))
                        })?
                };
                let input_b_bounds = if input_b_name == NETWORK_INPUT {
                    input
                } else {
                    crown_bounds
                        .get(input_b_name)
                        .or_else(|| ibp_bounds.get(input_b_name))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "MulBinary input B '{}' not found",
                                input_b_name
                            ))
                        })?
                };

                match mul.propagate_linear_binary(
                    &node_lb,
                    input_a_bounds,
                    input_b_bounds,
                    MulBinaryRelaxationMode::default(),
                ) {
                    Ok((mut lb_a, mut lb_b)) => {
                        debug!("{}: MulBinary '{}' CROWN succeeded", label, node_name);
                        let bias_lower = lb_a.lower_b() + lb_b.lower_b();
                        let bias_upper = lb_a.upper_b() + lb_b.upper_b();
                        lb_a.lower_b_mut().fill(0.0);
                        lb_a.upper_b_mut().fill(0.0);
                        lb_b.lower_b_mut().fill(0.0);
                        lb_b.upper_b_mut().fill(0.0);
                        Self::verify_split_path_bias_zero(
                            &lb_a,
                            &format!("{} MulBinary lhs split path", label),
                        )?;
                        Self::verify_split_path_bias_zero(
                            &lb_b,
                            &format!("{} MulBinary rhs split path", label),
                        )?;
                        Self::accumulate_bias_to_network_input_crown_with_deadline(
                            &bias_lower,
                            &bias_upper,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                            per_node_deadline,
                        )?;
                        self.accumulate_dense_bounds_to_input_with_deadline(
                            input_a_name,
                            lb_a,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                            per_node_deadline,
                        )?;
                        self.accumulate_dense_bounds_to_input_with_deadline(
                            input_b_name,
                            lb_b,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                            per_node_deadline,
                        )?;
                    }
                    Err(
                        e @ NyError::UnsupportedOp(_)
                        | e @ NyError::UnsupportedConfiguration(_)
                        | e @ NyError::NumericalInstability(_),
                    ) => {
                        debug!(
                            "{}: MulBinary '{}' failed ({}), returning UnsupportedOp for IBP fallback",
                            label, node_name, e
                        );
                        return Err(NyError::UnsupportedOp(format!(
                            "{}: MulBinary '{}' CROWN failed: {}",
                            label, node_name, e
                        )));
                    }
                    Err(e) => return Err(e),
                }
            } else if matches!(&node.layer, Layer::Div(_)) {
                let (input_a_name, input_b_name) = node.require_binary_inputs()?;
                let input_a_bounds = if input_a_name == NETWORK_INPUT {
                    input
                } else {
                    crown_bounds
                        .get(input_a_name)
                        .or_else(|| ibp_bounds.get(input_a_name))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Div numerator '{}' not found",
                                input_a_name
                            ))
                        })?
                };
                let input_b_bounds = if input_b_name == NETWORK_INPUT {
                    input
                } else {
                    crown_bounds
                        .get(input_b_name)
                        .or_else(|| ibp_bounds.get(input_b_name))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Div denominator '{}' not found",
                                input_b_name
                            ))
                        })?
                };
                let node_output_bounds = crown_bounds
                    .get(node_name)
                    .or_else(|| ibp_bounds.get(node_name))
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "Div output '{}' not found in bound maps",
                            node_name
                        ))
                    })?;
                match div::backward_div_to_numerator(
                    node_name,
                    &node_lb,
                    input_a_bounds,
                    input_b_bounds,
                    node_output_bounds,
                )? {
                    div::DivBackwardResult::PropagateNumerator(bounds) => {
                        self.accumulate_dense_bounds_to_input_with_deadline(
                            input_a_name,
                            *bounds,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                            per_node_deadline,
                        )?;
                    }
                    div::DivBackwardResult::ConcretizeCurrentNode { lower, upper } => {
                        Self::accumulate_bias_to_network_input_crown_with_deadline(
                            &lower,
                            &upper,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                            per_node_deadline,
                        )?;
                    }
                }
            } else if let Layer::Where(where_layer) = &node.layer {
                // === Embedded-constant Where (single `cond` input; both branches
                // constants). Output is a constant vector w.r.t. the network input;
                // fold it into the bias and route zero to `cond`. Exact per-element
                // select when `cond` is constant, sound IBP union otherwise.
                // require_ternary_inputs would error: the node has only 1 input.
                if where_layer.has_embedded_constants() {
                    let cond_input = node.require_unary_input()?;
                    let cond_bounds = if cond_input == NETWORK_INPUT {
                        input
                    } else {
                        crown_bounds
                            .get(cond_input)
                            .or_else(|| ibp_bounds.get(cond_input))
                            .ok_or_else(|| {
                                NyError::InvalidSpec(format!(
                                    "Where condition '{}' not found",
                                    cond_input
                                ))
                            })?
                    };
                    let select = where_layer.embedded_constant_select_output(cond_bounds)?;
                    let concrete =
                        node_lb.concretize_checked_with_deadline(&select, per_node_deadline)?;
                    let (lower, upper) =
                        concrete_into_ix1(concrete, "graph-alpha embedded-constant Where")?;
                    Self::accumulate_bias_to_network_input_crown_with_deadline(
                        &lower,
                        &upper,
                        &mut node_crown_bounds,
                        target_dim,
                        input_dim,
                        &mut input_accumulated,
                        per_node_deadline,
                    )?;
                    continue;
                }

                let (cond_input, true_input, false_input) = node.require_ternary_inputs()?;
                let cond_bounds = if cond_input == NETWORK_INPUT {
                    input
                } else {
                    crown_bounds
                        .get(cond_input)
                        .or_else(|| ibp_bounds.get(cond_input))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Where condition '{}' not found",
                                cond_input
                            ))
                        })?
                };
                let where_bounds = crown_bounds
                    .get(node_name)
                    .or_else(|| ibp_bounds.get(node_name))
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "Where output '{}' not found in bound maps",
                            node_name
                        ))
                    })?;
                let cond_all_true = cond_bounds.lower().iter().all(|&v| v >= 0.5);
                let cond_all_false = cond_bounds.upper().iter().all(|&v| v <= 0.5);

                if cond_all_true {
                    self.accumulate_dense_bounds_to_input_with_deadline(
                        true_input,
                        node_lb,
                        &mut node_crown_bounds,
                        target_dim,
                        input_dim,
                        &mut input_accumulated,
                        per_node_deadline,
                    )?;
                } else if cond_all_false {
                    self.accumulate_dense_bounds_to_input_with_deadline(
                        false_input,
                        node_lb,
                        &mut node_crown_bounds,
                        target_dim,
                        input_dim,
                        &mut input_accumulated,
                        per_node_deadline,
                    )?;
                } else if let Some(mask) =
                    crate::network::core::graph::backward_helpers::where_constant_mask(cond_bounds)
                {
                    // === Exact per-element select for a bound-independent (constant)
                    // condition mask. When `cond` is fixed (lower == upper), Where is
                    // a fixed 0/1 select: out[i] = true_input[i] if mask[i] else
                    // false_input[i] — an EXACT linear transform. Route each output
                    // column to the correct branch by zeroing the other branch's
                    // columns. Mirrors the graph-CROWN path; tighter than the loose
                    // concretize fallback below. Falls back to concretize on a shape
                    // mismatch (defensive; mask length == node_lb columns by
                    // construction).
                    if mask.len() == node_lb.num_inputs() {
                        let true_lb =
                            crate::network::core::graph::backward_helpers::mask_linear_bounds_columns(
                                &node_lb, &mask, true,
                            );
                        let false_lb =
                            crate::network::core::graph::backward_helpers::mask_linear_bounds_columns(
                                &node_lb, &mask, false,
                            );
                        self.accumulate_dense_bounds_to_input_with_deadline(
                            true_input,
                            true_lb,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                            per_node_deadline,
                        )?;
                        self.accumulate_dense_bounds_to_input_with_deadline(
                            false_input,
                            false_lb,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                            per_node_deadline,
                        )?;
                    } else {
                        let concrete = node_lb
                            .concretize_checked_with_deadline(where_bounds, per_node_deadline)?;
                        let (lower, upper) =
                            concrete_into_ix1(concrete, "graph-alpha Where mixed fallback")?;
                        Self::accumulate_bias_to_network_input_crown_with_deadline(
                            &lower,
                            &upper,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                            per_node_deadline,
                        )?;
                    }
                } else {
                    let concrete = node_lb
                        .concretize_checked_with_deadline(where_bounds, per_node_deadline)?;
                    let (lower, upper) =
                        concrete_into_ix1(concrete, "graph-alpha Where mixed fallback")?;
                    Self::accumulate_bias_to_network_input_crown_with_deadline(
                        &lower,
                        &upper,
                        &mut node_crown_bounds,
                        target_dim,
                        input_dim,
                        &mut input_accumulated,
                        per_node_deadline,
                    )?;
                }
            } else {
                // Strict finite-boundary dispatch is intentionally O(1) before
                // operator admission. Do not build the legacy combined map (or
                // clone its tensors) on that route; the strict dispatcher only
                // admits allocation-free control outcomes. Ordinary Dense
                // dispatch retains the exact historical merged lookup map.
                let combined_bounds = if finite_structured_boundary {
                    None
                } else {
                    let mut combined = std::collections::HashMap::new();
                    for inp_name in &node.inputs {
                        if inp_name == NETWORK_INPUT {
                            continue;
                        }
                        if let Some(b) = crown_bounds.get(inp_name) {
                            combined.insert(inp_name.clone(), b.clone());
                        } else if let Some(b) = ibp_bounds.get(inp_name) {
                            combined.insert(inp_name.clone(), b.clone());
                        }
                    }
                    // #cgan-coeff-err-fold: also expose THIS node's own output box.
                    // `dispatch_backward_layer` folds any incoming certified
                    // coefficient error into the bias via
                    // `ctx.node_bounds.get(ctx.node_name)` when the layer is not an
                    // err-carrier (BatchNorm, pooling, ...). Without the node's own
                    // bounds that fold degraded EVERY err-carrying row to
                    // `[-inf, +inf]` (`discharge_coeff_err_to_conservative`), which
                    // collapsed the whole per-target CROWN backward to IBP on
                    // conv->BatchNorm stacks (cGAN generators: conv backward attaches
                    // a fresh gamma_n*S err, the following BatchNorm found no output
                    // box, and the target bound became vacuous before the IBP
                    // intersection masked it as "Crown"). Sound: the fold is the
                    // established precise discharge over the node's certified output
                    // enclosure; providing the box only replaces the +/-inf degrade.
                    if let Some(b) = crown_bounds
                        .get(node_name)
                        .or_else(|| ibp_bounds.get(node_name))
                    {
                        combined.insert(node_name.clone(), b.clone());
                    }
                    Some(combined)
                };
                let dispatch_node_bounds = match combined_bounds.as_ref() {
                    Some(combined) => combined.into(),
                    // The strict route never reads this view before producing
                    // its allocation-free outcome; provide an existing map so
                    // admission itself requires no synthesized allocation.
                    None => ibp_bounds.into(),
                };

                let ctx = DispatchContext {
                    node_name,
                    layer: &node.layer,
                    inputs: &node.inputs,
                    pre_activation,
                    network_input: input,
                    node_bounds: dispatch_node_bounds,
                    engine,
                    deadline: per_node_deadline,
                    bilinear_alphas: None,
                    mul_binary_relaxation: MulBinaryRelaxationMode::default(),
                    mul_binary_alphas: None,
                    norm_inv_rms_override: None,
                };
                let result = if finite_structured_boundary {
                    dispatch_backward_layer_finite_boundary(&ctx, &node_lb)?
                } else {
                    dispatch_backward_layer(&ctx, &node_lb)?
                };
                match apply_dense_backward_dispatch_result_with_deadline(
                    self,
                    node,
                    first_input_name,
                    &node_lb,
                    result,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                    "Alpha-CROWN/IBP",
                    per_node_deadline,
                ) {
                    Ok(()) => {}
                    Err(NyError::UnsupportedOp(reason)) => {
                        return Err(NyError::UnsupportedOp(format!(
                            "{}: unsupported layer '{}' ({}): {}",
                            label,
                            node_name,
                            node.layer.layer_type(),
                            reason,
                        )));
                    }
                    Err(e) => return Err(e),
                }
            }
            if pass_probe {
                step_costs.push((
                    node_name.clone(),
                    step_shape.0.to_string(),
                    step_shape.1,
                    step_shape.2,
                    step_start.elapsed().as_secs_f64(),
                ));
            }
        }
        if pass_probe {
            let total = pass_start.elapsed().as_secs_f64();
            let mut by_kind: std::collections::BTreeMap<String, (usize, f64)> =
                std::collections::BTreeMap::new();
            for (_, kind, _, _, t) in &step_costs {
                let e = by_kind.entry(kind.clone()).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += t;
            }
            let mut worst = step_costs.clone();
            worst.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<String> = worst
                .iter()
                .take(4)
                .map(|(n, k, r, w, t)| format!("{n}[{k} r={r} w={w}]={t:.2}s"))
                .collect();
            let kinds: Vec<String> = by_kind
                .iter()
                .map(|(k, (n, t))| format!("{k}x{n}={t:.2}s"))
                .collect();
            eprintln!(
                "[phase] pass-cost target={target_node} steps={} total={total:.2}s {} | top: {}",
                step_costs.len(),
                kinds.join(" "),
                top.join(" ")
            );
        }

        if input_accumulated {
            let final_lb = node_crown_bounds
                .take_with_deadline(NETWORK_INPUT, per_node_deadline)?
                .ok_or_else(|| NyError::InvalidSpec("No linear bounds at input".to_string()))?;
            if capture_input_linear {
                if per_node_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
                    return Err(NyError::DeadlineExceeded(format!(
                        "{label}: per-node deadline exceeded before linear capture for \
                         target '{target_node}'"
                    )));
                }
                // The caller preflighted the complete dense A+bias+error
                // payload. `into_dense` preserves Patches coefficient-error
                // provenance; no concretization or error discharge happens
                // here.
                return Ok(TargetBackwardPassResult::InputLinear(
                    final_lb.into_dense_with_deadline_for_purpose(
                        per_node_deadline,
                        PatchesMaterializationPurpose::NetworkInputTerminal,
                    )?,
                ));
            }
            // #patches-row-range: a Patches-carried final relation whose dense
            // pair exceeds the CPU dense budget must never densify in one shot
            // — VGG16 conv targets reach 3.2M x 150K rows (~1.9 TB per matrix)
            // and abort the process. Per-row independence makes the blockwise
            // materialize-and-concretize bit-identical to the single-shot
            // path with memory bounded by the block budget; the deadline is
            // re-checked between blocks so a slow target degrades to the same
            // sound DeadlineExceeded fallback as the walk itself. The
            // under-budget path (including its DEBUG err diagnostic) is
            // unchanged byte-for-byte.
            let bounds = match final_lb {
                CrownBounds::Patches(pb)
                    if pb
                        .dense_pair_bytes()
                        .is_ok_and(|bytes| bytes > cpu_crown_dense_budget_bytes()) =>
                {
                    // #patches-sparse-concretize: prefer the patches-native sparse
                    // concretize — it visits only each row's receptive-field taps
                    // (~27 for a VGG16 conv1 target) instead of all 150528 input
                    // columns, turning the ~4.8e11-element dense-chunked traversal
                    // (a timeout) into ~Σ receptive-field work while staying
                    // BIT-IDENTICAL to `to_dense()?.concretize_sound(input)`. On an
                    // unsupported layout it returns `UnsupportedOp`; only then do we
                    // fall back to the certified (sound, memory-bounded) dense-chunked
                    // path. `DeadlineExceeded` and malformed-layout errors propagate.
                    match pb.concretize_sound_sparse(input, per_node_deadline) {
                        Ok(bounds) => bounds,
                        Err(NyError::UnsupportedOp(_)) => {
                            let block_bytes = cpu_crown_dense_budget_bytes()
                                .min(PATCHES_CONCRETIZE_MAX_BLOCK_BYTES);
                            pb.concretize_sound_chunked(input, block_bytes, per_node_deadline)?
                        }
                        Err(e) => return Err(e),
                    }
                }
                final_lb => {
                    let final_dense = final_lb.into_dense_with_deadline_for_purpose(
                        per_node_deadline,
                        PatchesMaterializationPurpose::NetworkInputTerminal,
                    )?;
                    if per_node_deadline.is_none() && tracing::enabled!(tracing::Level::DEBUG) {
                        // Diagnostic (#cgan-conv-err-compose): report the certified-error
                        // share of the final concretized width for this target.
                        let flat = input.flatten();
                        let xl = flat.lower();
                        let xu = flat.upper();
                        let mut max_pen = 0.0f64;
                        for err in [final_dense.lower_a_err(), final_dense.upper_a_err()]
                            .into_iter()
                            .flatten()
                        {
                            for i in 0..err.nrows() {
                                let mut pen = 0.0f64;
                                for j in 0..err.ncols() {
                                    let mag = (xl[j].abs()).max(xu[j].abs()) as f64;
                                    pen += err[[i, j]] as f64 * mag;
                                }
                                max_pen = max_pen.max(pen);
                            }
                        }
                        debug!(
                            "target '{}': final concretize max per-row err penalty {:.3e}",
                            target_node, max_pen
                        );
                    }
                    final_dense.concretize_sound_with_deadline(input, per_node_deadline)?
                }
            };
            Ok(TargetBackwardPassResult::Concrete(
                target_contract.restore_concrete_with_deadline(
                    bounds,
                    "Graph alpha-CROWN target restore",
                    per_node_deadline,
                )?,
            ))
        } else {
            // No input contribution accumulated; the caller substitutes the
            // pass-through target bounds (full target for single-pass, or the
            // chunk slice for objective-chunking).
            Ok(TargetBackwardPassResult::NoInputContribution)
        }
    }

    /// PRIMARY (#patches-obj-chunk): objective-chunking streaming driver.
    ///
    /// Streams the objective dimension (`target_dim` rows) in chunks of at most
    /// `chunk_size`. For each chunk `r0..r1`:
    ///   1. seed a `chunk_rows`-row slice of the objective identity (Dense rows
    ///      for a strict subset, or the canonical virtual Patches identity when
    ///      the one chunk covers the complete spatial output grid),
    ///   2. run the EXISTING backward loop (`run_target_backward_pass`) with a
    ///      per-chunk accumulator and a flat `[chunk_rows]` restore contract,
    ///   3. scatter the concretized `chunk_rows` values into the pre-sized
    ///      output, then drop the chunk coefficients before the next chunk.
    ///
    /// Bound-equivalent to the single pass by row-independence: the conv col2im
    /// scatter, the per-row CROWN error term, and the per-row concretize are all
    /// row-local, so concretizing rows `r0..r1` in isolation yields the same
    /// values they would take in a full-objective pass. The `GpuSuffixPlan` is
    /// objective-independent and is built once by the caller and reused here.
    ///
    /// `chunk_size` is the STARTING width and `chunk_ceiling` the widest chunk
    /// the caller admits (a memory bound when the caller asked for chunking;
    /// the full objective when only the deadline-granularity cap forced it).
    /// The deadline-bearing sequential driver widens between chunks
    /// (#chunk-grow) and abandons a walk whose projected finish passes the
    /// per-node deadline (#chunk-abort). Row-independence makes the partition
    /// itself unobservable in the bounds — widening changes only how many times
    /// the ancestor prefix is re-walked.
    #[allow(clippy::too_many_arguments)]
    fn propagate_crown_to_node_chunked(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        deadline_is_hard: bool,
        collector_patches_override: bool,
        relevant_nodes: &[String],
        target_bounds: &BoundedTensor,
        target_contract: &GraphTargetShapeContract,
        target_dim: usize,
        input_dim: usize,
        allow_patches: bool,
        gpu_suffix_plan: &GpuSuffixPlan,
        chunk_size: usize,
        chunk_ceiling: usize,
        cut_ctx: Option<&CrownCutContext>,
        salvage_deadline_chunks: bool,
        expected_fixed_waves: Option<ObjectiveChunkFixedWavePlan>,
    ) -> Result<TargetCrownCollectionResult> {
        if let Some(limit) = per_node_deadline {
            if std::time::Instant::now() >= limit {
                return Err(NyError::DeadlineExceeded(format!(
                    "{label}: objective-chunk deadline expired before admission for target \
                     '{target_node}'"
                )));
            }
        }
        if deadline_is_hard {
            // Hard-deadline chunking still performs whole-target setup before its
            // between-chunk polls: endpoint flatten/copy, range-vector
            // allocation, GPU-suffix planning, and dense subset-seed
            // allocation/fill. Refuse at the entry boundary until those phases
            // are cooperative. Small finite targets that do not require this
            // driver continue through the ordinary non-chunked path; `None`
            // retains the historical objective-chunk evaluator exactly.
            return Err(NyError::UnsupportedConfiguration(format!(
                "{label}: finite objective-chunk target backward is not cooperatively bounded \
                 for target '{target_node}'"
            )));
        }

        // The Patches seed is only valid for a 3D spatial objective; otherwise
        // (and whenever `allow_patches` is false) seed Dense identity rows.
        let target_shape = target_bounds.shape();
        let patches_seed = allow_patches && target_shape.len() == 3;
        let spatial = if patches_seed {
            Some((target_shape[0], target_shape[1], target_shape[2]))
        } else {
            None
        };

        // Flat target box used both for pass-through chunks and as the certified
        // seed for rows a deadline-truncated collector never completes.
        let target_flat = target_bounds.flatten();
        let target_lower_flat = target_flat.lower();
        let target_upper_flat = target_flat.upper();

        // The chunk ranges `[r0, r1)` partition the objective rows.
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut r0 = 0usize;
        while r0 < target_dim {
            let r1 = (r0 + chunk_size).min(target_dim);
            ranges.push((r0, r1));
            r0 = r1;
        }

        // Bound one chunk `[r0, r1)`. ROW-INDEPENDENT (see this fn's docstring): the
        // col2im scatter, the per-row CROWN error term, and the per-row concretize are
        // all row-local, so a chunk's rows take the same values in isolation as in the
        // full-objective pass. Returns `(r0, lower_rows, upper_rows)`.
        let bound_chunk = |&(r0, r1): &(usize, usize),
                           ctx: Option<&CrownCutContext>|
         -> Result<CompletedCrownChunk> {
            let chunk_rows = r1 - r0;
            let seed = build_chunk_seed(target_dim, r0, r1, spatial, per_node_deadline)?;
            // Flat restore contract for this chunk: 1D `[chunk_rows]`, so the
            // GPU-suffix / final concretize restore is an identity reshape and
            // the produced bounds line up 1:1 with the chunk's output rows.
            let chunk_contract =
                GraphTargetShapeContract::from_bounds(target_node, &flat_bounds_view(chunk_rows)?);

            let produced = self.run_target_backward_pass(
                input,
                target_node,
                crown_bounds,
                ibp_bounds,
                alpha_state,
                engine,
                label,
                per_node_deadline,
                deadline_is_hard,
                collector_patches_override,
                relevant_nodes,
                &chunk_contract,
                chunk_rows,
                input_dim,
                allow_patches,
                gpu_suffix_plan,
                seed,
                ctx,
            )?;

            // A layer kernel may cross the deadline before it can return to
            // `run_target_backward_pass`'s per-node poll.  Check immediately
            // after every bounded objective chunk, including the final chunk,
            // so a late last chunk cannot be accepted as an on-time result.
            if per_node_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
                return Err(NyError::DeadlineExceeded(format!(
                    "{label}: per-node deadline exceeded after objective chunk \
                     {r0}..{r1} for target '{target_node}'"
                )));
            }

            match produced {
                Some(bounds) => {
                    let lo = bounds.lower();
                    let up = bounds.upper();
                    let lo = lo.as_slice().ok_or_else(|| {
                        NyError::InvalidSpec(
                            "objective-chunk concrete lower not contiguous".to_string(),
                        )
                    })?;
                    let up = up.as_slice().ok_or_else(|| {
                        NyError::InvalidSpec(
                            "objective-chunk concrete upper not contiguous".to_string(),
                        )
                    })?;
                    Ok((r0, lo.to_vec(), up.to_vec()))
                }
                None => {
                    // No input contribution accumulated for this chunk: pass the
                    // target bounds slice through unchanged (mirrors the single-
                    // pass `target_bounds.clone()` terminal branch, restricted to
                    // these rows).
                    let lo: Vec<f32> = (r0..r1).map(|row| target_lower_flat[[row]]).collect();
                    let up: Vec<f32> = (r0..r1).map(|row| target_upper_flat[[row]]).collect();
                    Ok((r0, lo, up))
                }
            }
        };

        let finish_deadline = |completed_chunks: Vec<CompletedCrownChunk>,
                               error: NyError|
         -> Result<TargetCrownCollectionResult> {
            match error {
                NyError::DeadlineExceeded(details) if salvage_deadline_chunks => {
                    let completed_rows = completed_chunks
                        .iter()
                        .map(|(_, lower, _)| lower.len())
                        .sum::<usize>();
                    if completed_rows > 0 {
                        info!(
                            "[NY_CROWN_DEADLINE_CHUNK_SALVAGE_V1] stage=retained \
                             target='{target_node}' completed_rows={completed_rows} \
                             total_rows={target_dim} committed_chunks={}",
                            completed_chunks.len(),
                        );
                    } else {
                        debug!(
                            "[NY_CROWN_DEADLINE_CHUNK_SALVAGE_V1] stage=deadline-no-rows \
                             target='{target_node}' total_rows={target_dim}"
                        );
                    }
                    assemble_completed_crown_chunks(
                        target_bounds,
                        target_contract,
                        target_dim,
                        completed_chunks,
                        Some(details),
                    )
                }
                error => Err(error),
            }
        };

        // PARALLEL chunk driver — opt-in via the IMB anchor scope
        // (`crate::imb::anchor_chunk_parallel`). The chunks are row-independent, so a
        // parallel evaluation is bound-equivalent AND order-independent (each writes a
        // DISJOINT `out_*[r0..r1]` range) ⇒ deterministic. Faer stays Seq-guarded
        // inside the rayon workers (`current_par`), so the parallelism is across
        // chunks, not nested. Every OTHER caller (the collector's over-budget chunking
        // included) keeps the sequential loop below → byte-identical.
        // The cut-segment context (`NY_CROWN_CUT_SEGMENT`) carries interior
        // mutability (`Cell` hit counters) and is not `Sync`; every parallel
        // caller (the IMB anchor scope) passes `cut_ctx: None`, so ctx-carrying
        // sweeps simply keep the sequential branch. Deadline-bearing chunks
        // are also sequential: launching all ranges at once would defeat the
        // bounded-work guarantee and the between-chunk deadline poll.
        let chunk_out: Vec<CompletedCrownChunk> = {
            //
            // #chunk-wave-par: a per-node DEADLINE no longer forces the sequential
            // driver. It used to, and that is why this parallel branch never ran
            // for the CROWN-IBP collector -- which always sets a per-node deadline.
            // The consequence was measured on CIFAR100_resnet_medium: ny reported
            // "8/11 DEMANDED targets reverted to IBP (PerNodeDeadlineExceeded=8)"
            // while ~95% of a 32-thread machine sat in rayon's idle path. The
            // sequential driver's answer to a target that cannot finish in budget
            // is to ABORT it early (#chunk-abort) -- optimal only if going faster
            // is impossible, which it plainly was not with 30 cores parked.
            //
            // A deadline-bearing sweep now runs its chunks in parallel WAVES: each
            // wave is a bounded amount of work, and the deadline poll and the
            // projection abort both happen BETWEEN waves. That keeps exactly the
            // two properties the sequential driver existed to provide (bounded
            // unpolled work, early abort of a walk that cannot land) while using
            // the idle cores. Output is unchanged: the ranges partition the
            // objective rows, each chunk writes a DISJOINT `out_*[r0..r1]`, and row
            // values are partition-invariant, so the result is bound-identical and
            // deterministic regardless of wave scheduling.
            // `NY_NO_CHUNK_WAVE_PAR=1` restores the sequential deadline driver.
            //
            // MEASURED BENEFIT: NONE YET, on CIFAR100_resnet_medium. A/B at a 300s
            // budget with the CROWN-IBP collector forced live
            // (NY_NO_FORWARD_LINEAR_REF=1) gave 186.8s sequential vs 185.3s wave --
            // indistinguishable. The change is kept because the gate it removes is
            // wrong in principle -- a deadline should not force serial execution
            // while the machine is ~95% idle -- and because it is bound-identical
            // and covered by the chunk-equivalence proptests, NOT because it has
            // been shown to help. Do not cite it as a speedup.
            //
            // The two candidate explanations above were "targets partition into a
            // single chunk, so the wave path is skipped" vs "the collector's cost
            // is elsewhere". SETTLED, 2026-08-01, in favour of the SECOND: forcing
            // a multi-chunk partition does not reduce reverts, it increases them.
            // cifar100_2024 rows 1..4, official 100 s budget, RTX 5080 / 10-CPU
            // cgroup, counting DEMANDED targets reverted to IBP:
            //
            //   NY_CROWN_OBJ_CHUNK=0 (default)  21/33 reverted   solved 1/4
            //   NY_CROWN_OBJ_CHUNK=512          24/33            solved 1/4
            //   NY_CROWN_OBJ_CHUNK=256          24/33            solved 1/4
            //   NY_CROWN_OBJ_CHUNK=64           26/33            solved 1/4
            //
            // Monotone in the wrong direction: narrower chunks cost more than the
            // parallelism they buy at this width, so the unchunked default is
            // already the better partition and the wave driver is not being starved
            // of chunks. Whatever dominates the collector's wall on this network is
            // upstream of the objective-row partition. Do not re-run this sweep.
            let driver_route = enforce_expected_fixed_wave_plan(
                live_objective_chunk_driver_route(
                    target_dim,
                    chunk_size,
                    chunk_ceiling,
                    per_node_deadline.is_some(),
                    cut_ctx.is_some(),
                ),
                expected_fixed_waves,
                label,
                target_node,
            )?;
            if let ObjectiveChunkDriverRoute::FixedWaves(wave_plan) = driver_route {
                use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
                let deadline =
                    per_node_deadline.expect("fixed-wave route requires a per-node deadline");
                debug_assert_eq!(ranges.len(), wave_plan.chunk_count);
                let abort_enabled_for_wave = chunk_projection_abort_enabled();

                let mut bounded: Vec<(usize, Vec<f32>, Vec<f32>)> =
                    Vec::with_capacity(ranges.len());
                let mut spent = std::time::Duration::ZERO;
                let mut rows_done = 0usize;
                for chunk_of_ranges in ranges.chunks(wave_plan.wave_size) {
                    if let Some(error) = injected_chunk_controller_deadline(bounded.len()) {
                        return finish_deadline(bounded, error);
                    }
                    if std::time::Instant::now() >= deadline {
                        let error = NyError::DeadlineExceeded(format!(
                            "{label}: per-node deadline reached during the objective-chunk \
                         wave walk for target '{target_node}' ({rows_done}/{target_dim} rows)"
                        ));
                        return finish_deadline(bounded, error);
                    }
                    let wave_start = std::time::Instant::now();
                    // Commit a wave atomically. If any member reports a typed
                    // deadline, every result from this in-flight wave is rejected;
                    // only earlier, fully committed waves may be salvaged.
                    let got_result: Result<Vec<CompletedCrownChunk>> = chunk_of_ranges
                        .par_iter()
                        .map(|r| bound_chunk(r, None))
                        .collect();
                    let mut got = match got_result {
                        Ok(got) => got,
                        Err(error) => return finish_deadline(bounded, error),
                    };
                    // Worker-local post-chunk polls are not sufficient authority:
                    // every worker can finish before the deadline while the rayon
                    // join/coordinator handoff crosses it. Keep this wave
                    // provisional until one shared post-join poll succeeds. A
                    // late wave is rejected atomically, including the final wave,
                    // before append or Complete classification can publish it.
                    if chunk_post_join_now(deadline, bounded.len()) >= deadline {
                        let error = NyError::DeadlineExceeded(format!(
                            "{label}: per-node deadline reached after objective-chunk \
                             wave join and before commit for target '{target_node}' \
                             ({rows_done}/{target_dim} rows previously committed)"
                        ));
                        return finish_deadline(bounded, error);
                    }
                    let wave_elapsed = wave_start.elapsed();
                    spent += wave_elapsed;
                    let wave_rows: usize = chunk_of_ranges.iter().map(|(a, b)| b - a).sum();
                    rows_done += wave_rows;
                    bounded.append(&mut got);
                    if rows_done >= target_dim {
                        break;
                    }

                    // Same #chunk-abort projection, measured per WAVE: a walk that
                    // cannot land inside the per-node budget should stop now.
                    // An armed collector retains prior committed waves over
                    // IBP; default-dark and strict callers receive the same
                    // typed deadline as before.
                    if abort_enabled_for_wave {
                        let rows_left = target_dim - rows_done;
                        let rate_per_row = wave_elapsed.as_secs_f64() / wave_rows.max(1) as f64;
                        let left = deadline
                            .saturating_duration_since(std::time::Instant::now())
                            .as_secs_f64();
                        let projected_s = rate_per_row * rows_left as f64;
                        if projected_s > left * CHUNK_ABORT_SLACK {
                            info!(
                            "{label}: target '{target_node}' objective-chunk WAVE walk aborted \
                             early (#chunk-abort): {rows_done} of {target_dim} rows in {:.1}s, \
                             projected {projected_s:.1}s for the remaining {rows_left} rows \
                             with {left:.1}s left — ending the truncated walk now",
                            spent.as_secs_f64(),
                        );
                            let error = NyError::DeadlineExceeded(format!(
                                "{label}: projected objective-chunk wave finish for target \
                             '{target_node}' exceeds the per-node deadline \
                             ({rows_done}/{target_dim} rows)"
                            ));
                            return finish_deadline(bounded, error);
                        }
                    }
                }
                bounded
            } else if matches!(driver_route, ObjectiveChunkDriverRoute::AnchorParallel) {
                use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
                // `ctx = None` at compile level: the cut context is `Cell`-bearing
                // (not `Sync`), and every parallel caller passes `None` anyway
                // (enforced by the `cut_ctx.is_none()` gate above).
                ranges
                    .par_iter()
                    .map(|r| bound_chunk(r, None))
                    .collect::<Result<Vec<_>>>()?
            } else {
                // SEQUENTIAL (deadline-bearing) driver: adaptive chunk width
                // (#chunk-grow) plus the measured-projection early abort
                // (#chunk-abort). A chunk's per-ROW cost is stable across widths
                // (measured on cgan_2023 ConvTranspose_7: 0.037 s/row at width 32
                // and again at width 128), so the last chunk's rate projects the
                // remaining rows accurately; as soon as that projection passes the
                // per-node deadline the walk CANNOT deliver this target and every
                // further chunk is dead work. An armed collector retains already
                // completed rows over IBP; default-dark and strict callers receive
                // the same typed deadline as before. In both cases the wasted
                // seconds are reclaimed for the remaining targets.
                // `NY_NO_CHUNK_ABORT=1` restores the burn-the-whole-budget
                // behavior; `NY_NO_CHUNK_GROW=1` restores the fixed-width partition.
                let abort_enabled = chunk_projection_abort_enabled();
                let ObjectiveChunkDriverRoute::Sequential {
                    adaptive_growth: grow_enabled,
                } = driver_route
                else {
                    unreachable!("non-sequential objective-chunk route reached sequential driver")
                };
                let mut bounded: Vec<(usize, Vec<f32>, Vec<f32>)> =
                    Vec::with_capacity(ranges.len());
                let mut spent = std::time::Duration::ZERO;
                let mut chunks_done = 0usize;
                let mut width = chunk_size;
                let mut start_row = 0usize;
                while start_row < target_dim {
                    if let Some(error) = injected_chunk_controller_deadline(chunks_done) {
                        return finish_deadline(bounded, error);
                    }
                    let end_row = (start_row + width).min(target_dim);
                    let chunk_start = std::time::Instant::now();
                    let chunk = match bound_chunk(&(start_row, end_row), cut_ctx) {
                        Ok(chunk) => chunk,
                        Err(error) => return finish_deadline(bounded, error),
                    };
                    bounded.push(chunk);
                    let chunk_elapsed = chunk_start.elapsed();
                    spent += chunk_elapsed;
                    chunks_done += 1;
                    let rows_bounded = (end_row - start_row).max(1);
                    let rate_per_row = chunk_elapsed.as_secs_f64() / rows_bounded as f64;
                    start_row = end_row;
                    if start_row >= target_dim {
                        break;
                    }
                    let rows_left = target_dim - start_row;

                    // #chunk-abort — projected-finish early abort. A walk that
                    // cannot finish inside the per-node budget is DEAD work: an
                    // explicitly armed collector preserves every completed row
                    // over its certified IBP seed and records explicit truncated
                    // provenance. Aborting as soon as the projection (the last
                    // chunk's measured per-row rate, with slack for the prefix
                    // amortization a wider next chunk gains) passes the deadline
                    // hands the unspent seconds to the remaining targets and BaB.
                    if abort_enabled && chunks_done >= CHUNK_ABORT_MIN_SAMPLES {
                        if let Some(deadline) = per_node_deadline {
                            let left = deadline
                                .saturating_duration_since(std::time::Instant::now())
                                .as_secs_f64();
                            let projected_s = rate_per_row * rows_left as f64;
                            if projected_s > left * CHUNK_ABORT_SLACK {
                                info!(
                                    "{label}: target '{target_node}' objective-chunk walk aborted \
                                 early (#chunk-abort): {chunks_done} chunks / {start_row} of \
                                 {target_dim} rows in {:.1}s, projected {projected_s:.1}s for \
                                 the remaining {rows_left} rows with {left:.1}s left — \
                                 ending the truncated walk now instead of burning the per-node budget",
                                    spent.as_secs_f64(),
                                );
                                // #cprime-abort-calib: publish the FULL-walk cost
                                // this abort just measured (spent + projected
                                // remainder). Where no walk ever completes —
                                // cgan_2023 — this is the ONLY rate sample c′ can
                                // calibrate from; without it the optimistic prior
                                // admits every doomed walk (measured there:
                                // admitted=61 refused=0).
                                budget_policy::publish_walk_abort_projection(
                                    spent.as_secs_f64() + projected_s,
                                );
                                let error = NyError::DeadlineExceeded(format!(
                                    "{label}: projected objective-chunk finish for target \
                                 '{target_node}' exceeds the per-node deadline \
                                 ({start_row}/{target_dim} rows in {:.1}s, projected \
                                 {projected_s:.1}s for the rest)",
                                    spent.as_secs_f64(),
                                ));
                                return finish_deadline(bounded, error);
                            }
                        }
                    }

                    // #chunk-grow — widen the next chunk toward the unpolled-work
                    // target. EVERY chunk re-walks the target's whole ancestor
                    // prefix, so a fixed 32-row chunk pays that prefix
                    // `ceil(dim/32)` times. Measured on cgan_2023
                    // (cGAN_imgSz32_nCh_1 prop_1, 900 s official budget, preset
                    // loaded): three collection targets each burned their entire
                    // 150 s per-node cap WITHOUT finishing, while
                    // the same collection measured 142.96 s TOTAL unchunked. Growth
                    // keeps one unpolled pass bounded by `CHUNK_GROW_TARGET_SECS`
                    // (the reason the 32-row deadline cap exists) while paying the
                    // prefix a handful of times instead of hundreds.
                    // BOUND-IDENTICAL by row-independence: only the partition of the
                    // objective rows changes, and every row's value is partition
                    // invariant (see this fn's docstring). `NY_NO_CHUNK_GROW=1`
                    // restores the fixed-width partition.
                    if grow_enabled && width < chunk_ceiling {
                        let target_rows = if rate_per_row > 0.0 {
                            ((CHUNK_GROW_TARGET_SECS / rate_per_row) as usize).max(1)
                        } else {
                            chunk_ceiling
                        };
                        let grown = target_rows
                            .max(width)
                            .min(width.saturating_mul(CHUNK_GROW_MAX_FACTOR))
                            .min(chunk_ceiling);
                        if grown > width {
                            width = grown;
                        }
                    }
                }
                bounded
            }
        };
        assemble_completed_crown_chunks(target_bounds, target_contract, target_dim, chunk_out, None)
    }

    /// #margin-subset-seed: selector-seeded k-row variant of the full-width
    /// target backward.
    ///
    /// Seeds the backward walk with the `rows.len()` identity rows named by
    /// `rows` (arbitrary flat output positions, not necessarily contiguous)
    /// instead of the full `[target_dim x target_dim]` identity, and returns
    /// the concretized `(lower, upper)` values for exactly those rows, in
    /// `rows` order. On vggnet16's 1000-wide OUTPUT node with a 2-index margin
    /// spec this turns the `[1000 x 401408]` conv coefficient buffers
    /// (~1.6 GiB each) into `[2 x 401408]` (~3.2 MiB).
    ///
    /// SOUNDNESS / ROW-EQUIVALENCE: each identity seed row is an independent
    /// linear objective — the conv col2im scatter, the per-row CROWN error
    /// term, and the per-row concretize are all row-local (the same
    /// row-independence the #patches-obj-chunk objective chunking relies on),
    /// so every returned row is bit-identical in semantics to the same row of
    /// the full-width backward under the same walk configuration. The seed
    /// builder (`build_subset_seed`) is shared with the chunk driver.
    ///
    /// "Under the same walk configuration" is load-bearing: for a SPATIAL,
    /// patches-eligible target the k-row seed is Dense (see
    /// [`build_subset_seed`]) while the full-width seed is a virtual Patches
    /// identity, so the two walks differ in REPRESENTATION and the rows are
    /// bound-equivalent rather than bit-equal (both are valid CROWN
    /// enclosures of the same linear objective; measured difference on the
    /// unit fixture is 2.8% of the interval width, in the looser direction).
    /// Row-independence within one configuration is unaffected, so the chunk
    /// partition itself remains unobservable.
    ///
    /// The intended caller is the CROWN-IBP collector's OUTPUT-node consume
    /// (crown_tighten.rs), which scatters these k rows over the node's sound
    /// IBP bounds and intersects with IBP exactly as for full maps; on ANY
    /// error from this method it falls back to the existing full-width path
    /// (fail-open to the old behavior). `alpha_state` is always `None` here,
    /// matching both collector entry points.
    #[allow(clippy::too_many_arguments)] // Mirrors propagate_crown_to_node_core's threading (#3549).
    pub(in crate::network::graph_alpha) fn propagate_crown_to_node_subset(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        deadline_is_hard: bool,
        collector_patches_override: bool,
        rows: &[usize],
        cut_ctx: Option<&CrownCutContext>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let target_bounds = ibp_bounds.get(target_node).ok_or_else(|| {
            NyError::InvalidSpec(format!("Target node {} not in IBP bounds", target_node))
        })?;
        let target_contract = GraphTargetShapeContract::from_bounds(target_node, target_bounds);
        let target_dim = target_contract.flat_dim();
        if rows.is_empty() {
            return Err(NyError::InvalidSpec(
                "margin-subset backward requires at least one seed row".to_string(),
            ));
        }
        if let Some(&bad) = rows.iter().find(|&&row| row >= target_dim) {
            return Err(NyError::InvalidSpec(format!(
                "margin-subset seed row {bad} out of range for target '{target_node}' \
                 (flat dim {target_dim})"
            )));
        }

        let relevant_nodes = self.ancestors(target_node)?;
        if relevant_nodes.is_empty() {
            // Full-width counterpart returns `input.clone()`; mirror it for
            // the requested rows (only meaningful when the target IS the
            // network input passthrough, so the dims must agree).
            let flat = input.flatten();
            if flat.len() != target_dim {
                return Err(NyError::InvalidSpec(format!(
                    "margin-subset target '{target_node}' has no ancestors and input dim {} \
                     != target dim {target_dim}",
                    flat.len()
                )));
            }
            let lower_flat = flat.lower();
            let upper_flat = flat.upper();
            let lower = rows.iter().map(|&row| lower_flat[[row]]).collect();
            let upper = rows.iter().map(|&row| upper_flat[[row]]).collect();
            return Ok((lower, upper));
        }

        if deadline_is_hard
            && !crate::sound_gpu_gate::gpu_crown_route_honors_deadline(engine, per_node_deadline)
        {
            return Err(NyError::SoundnessRefusal(
                "deadline-scored target backward requires a cooperative GPU backend".to_string(),
            ));
        }

        // Same cooperative GPU deadline scope as the full-width core
        // (#w4-refresh-deadline).
        let _gpu_deadline_scope =
            crate::sound_gpu_gate::GpuCrownDeadlineScope::set(engine, per_node_deadline);

        let allow_patches = target_allows_patches_start(
            self,
            target_node,
            None,
            &relevant_nodes,
            target_bounds,
            collector_patches_override,
        );
        let target_shape = target_bounds.shape();
        let spatial = if allow_patches && target_shape.len() == 3 {
            Some((target_shape[0], target_shape[1], target_shape[2]))
        } else {
            None
        };
        let seed = build_subset_seed(target_dim, rows, spatial, per_node_deadline)?;
        let gpu_suffix_plan =
            GpuSuffixPlan::build(&relevant_nodes, self, input, crown_bounds, ibp_bounds, None);
        let k = rows.len();
        // Flat [k] restore contract: the produced bounds line up 1:1 with `rows`.
        let subset_contract =
            GraphTargetShapeContract::from_bounds(target_node, &flat_bounds_view(k)?);

        let produced = self.run_target_backward_pass(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            None,
            engine,
            label,
            per_node_deadline,
            deadline_is_hard,
            collector_patches_override,
            relevant_nodes.as_slice(),
            &subset_contract,
            k,
            input.len(),
            allow_patches,
            &gpu_suffix_plan,
            seed,
            cut_ctx,
        )?;
        // A layer kernel may cross the deadline before returning to the
        // per-node poll; a late result must not be accepted as on-time.
        if per_node_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
            return Err(NyError::DeadlineExceeded(format!(
                "{label}: per-node deadline exceeded after margin-subset backward for \
                 target '{target_node}'"
            )));
        }
        match produced {
            Some(bounds) => {
                let lower = bounds.lower();
                let upper = bounds.upper();
                let lower = lower.as_slice().ok_or_else(|| {
                    NyError::InvalidSpec("margin-subset concrete lower not contiguous".to_string())
                })?;
                let upper = upper.as_slice().ok_or_else(|| {
                    NyError::InvalidSpec("margin-subset concrete upper not contiguous".to_string())
                })?;
                if lower.len() != k || upper.len() != k {
                    return Err(NyError::InvalidSpec(format!(
                        "margin-subset backward produced {} rows, expected {k}",
                        lower.len()
                    )));
                }
                Ok((lower.to_vec(), upper.to_vec()))
            }
            None => {
                // No input contribution accumulated: pass the target rows
                // through unchanged (mirrors the single-pass
                // `target_bounds.clone()` terminal branch, restricted to
                // these rows).
                let flat = target_bounds.flatten();
                let lower_flat = flat.lower();
                let upper_flat = flat.upper();
                let lower = rows.iter().map(|&row| lower_flat[[row]]).collect();
                let upper = rows.iter().map(|&row| upper_flat[[row]]).collect();
                Ok((lower, upper))
            }
        }
    }

    /// Premise-scoped output-conditioned two-node CROWN evaluator.
    ///
    /// This bounds selected coordinates of `target_node` under the premise
    /// `objective_row · output <= threshold`. The target identity and the
    /// non-negative premise dual are inserted at distinct graph nodes, then
    /// merged before traversing the target's ancestors. Returned endpoints are
    /// intersected with the inherited unconditional target box but are never
    /// written into a graph/root/BaB cache.
    ///
    /// The sole production caller keeps the returned box call-local and grants
    /// authority only after an ordinary same-row spec replay refutes the exact
    /// premise. A nonzero combined frontier disables the existing GPU suffix
    /// shortcuts because their seed representation cannot carry this
    /// evaluator's certified coefficient-error channel. All-zero gamma omits
    /// the output frontier entirely and takes the ordinary single-seed target
    /// path.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn propagate_output_conditioned_crown_to_node_subset(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        rows: &[usize],
        objective_row: &[f32],
        threshold: f32,
        gammas_lower: &[f32],
        gammas_upper: &[f32],
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(
                "output-conditioned two-seed backward deadline expired before admission"
                    .to_string(),
            ));
        }
        // The legacy two-frontier evaluator has several whole-request setup
        // phases that cannot observe a deadline: graph-order/ancestry cache
        // construction, dense output-seed allocation/fill, GPU-suffix planning,
        // and final inherited-bound flattening.  A live finite authority must
        // therefore decline before even validating/scanning the request.  The
        // sole production caller treats this optional treatment as a miss and
        // continues with the ordinary certified proof route.  `None` preserves
        // the exact historical evaluator.
        if deadline.is_some() {
            return Err(NyError::UnsupportedConfiguration(
                "finite output-conditioned two-seed backward is not cooperatively bounded"
                    .to_string(),
            ));
        }
        if rows.is_empty()
            || rows.len() > OUTPUT_CONDITIONED_TWO_SEED_MAX_ROWS
            || rows.len() != gammas_lower.len()
            || rows.len() != gammas_upper.len()
        {
            return Err(NyError::InvalidSpec(
                "output-conditioned two-seed row/gamma shape mismatch".to_string(),
            ));
        }
        let mut unique_rows = std::collections::HashSet::with_capacity(rows.len());
        if rows.iter().any(|row| !unique_rows.insert(*row)) {
            return Err(NyError::InvalidSpec(
                "output-conditioned two-seed rows must be unique".to_string(),
            ));
        }

        let target_bounds = ibp_bounds.get(target_node).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "output-conditioned target node '{target_node}' has no IBP bounds"
            ))
        })?;
        let target_width = target_bounds.len();
        if let Some(&bad) = rows.iter().find(|&&row| row >= target_width) {
            return Err(NyError::InvalidSpec(format!(
                "output-conditioned target row {bad} is outside width {target_width}"
            )));
        }

        let exec_order = self.exec_order()?;
        let output_node = if self.output_name().is_empty() {
            exec_order.last().cloned().ok_or_else(|| {
                NyError::InvalidSpec(
                    "output-conditioned two-seed backward requires a graph output".to_string(),
                )
            })?
        } else {
            self.output_name().to_string()
        };
        if output_node == target_node {
            return Err(NyError::InvalidSpec(
                "output-conditioned target must precede the graph output".to_string(),
            ));
        }
        let output_bounds = ibp_bounds.get(&output_node).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "output-conditioned output node '{output_node}' has no IBP bounds"
            ))
        })?;
        if objective_row.len() != output_bounds.len() {
            return Err(NyError::InvalidSpec(format!(
                "output-conditioned objective width {} does not match output width {}",
                objective_row.len(),
                output_bounds.len()
            )));
        }

        let output_relevant_nodes = self.ancestors(&output_node)?;
        if !output_relevant_nodes.iter().any(|name| name == target_node) {
            return Err(NyError::InvalidSpec(format!(
                "output-conditioned target '{target_node}' is not on the output ancestry"
            )));
        }
        if !crate::sound_gpu_gate::gpu_crown_route_honors_deadline(engine, deadline) {
            return Err(NyError::SoundnessRefusal(
                "deadline-scored output-conditioned backward requires a cooperative GPU backend"
                    .to_string(),
            ));
        }

        let output_seed = build_output_conditioned_output_seed(
            objective_row,
            threshold,
            gammas_lower,
            gammas_upper,
        )?;
        let zero_gamma = gammas_lower
            .iter()
            .chain(gammas_upper)
            .all(|gamma| *gamma == 0.0);
        let relevant_nodes = if zero_gamma {
            self.ancestors(target_node)?
        } else {
            output_relevant_nodes
        };
        let allow_patches = zero_gamma
            && target_allows_patches_start(
                self,
                target_node,
                alpha_state,
                &relevant_nodes,
                target_bounds,
                false,
            );
        let target_shape = target_bounds.shape();
        let target_spatial = if allow_patches && target_shape.len() == 3 {
            Some((target_shape[0], target_shape[1], target_shape[2]))
        } else {
            None
        };
        let target_seed = build_subset_seed(target_width, rows, target_spatial, deadline)?;
        let additional_seed = if zero_gamma {
            None
        } else {
            Some(AdditionalTargetBackwardSeed {
                node_name: &output_node,
                bounds: output_seed,
            })
        };
        let row_contract =
            GraphTargetShapeContract::from_bounds(target_node, &flat_bounds_view(rows.len())?);
        let gpu_suffix_plan = GpuSuffixPlan::build(
            &relevant_nodes,
            self,
            input,
            crown_bounds,
            ibp_bounds,
            alpha_state,
        );
        let _gpu_deadline_scope =
            crate::sound_gpu_gate::GpuCrownDeadlineScope::set(engine, deadline);
        let produced = self.run_target_backward_pass_core(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            engine,
            "output-conditioned two-seed CROWN",
            deadline,
            deadline.is_some(),
            false,
            relevant_nodes.as_slice(),
            &row_contract,
            rows.len(),
            input.len(),
            allow_patches,
            &gpu_suffix_plan,
            target_seed,
            None,
            additional_seed,
            false,
        )?;
        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(
                "output-conditioned two-seed backward deadline expired after evaluation"
                    .to_string(),
            ));
        }
        let conditioned = match produced {
            TargetBackwardPassResult::Concrete(bounds) => bounds,
            TargetBackwardPassResult::NoInputContribution => {
                return Err(NyError::SoundnessRefusal(
                    "output-conditioned two-seed backward produced no input contribution"
                        .to_string(),
                ));
            }
            TargetBackwardPassResult::InputLinear(_) => {
                return Err(NyError::InternalError(
                    "output-conditioned concrete backward returned an input-linear certificate"
                        .to_string(),
                ));
            }
        };
        let conditioned = conditioned.flatten();
        let conditioned_lower = conditioned.lower();
        let conditioned_upper = conditioned.upper();
        if conditioned_lower.len() != rows.len() || conditioned_upper.len() != rows.len() {
            return Err(NyError::InvalidSpec(
                "output-conditioned two-seed backward returned the wrong row count".to_string(),
            ));
        }

        let inherited = crown_bounds
            .get(target_node)
            .unwrap_or(target_bounds)
            .flatten();
        if inherited.len() != target_width {
            return Err(NyError::InvalidSpec(
                "output-conditioned inherited target box has the wrong width".to_string(),
            ));
        }
        let mut lower = Vec::with_capacity(rows.len());
        let mut upper = Vec::with_capacity(rows.len());
        for (position, &target_row) in rows.iter().enumerate() {
            let candidate_lower = conditioned_lower[[position]];
            let candidate_upper = conditioned_upper[[position]];
            let inherited_lower = inherited.lower()[[target_row]];
            let inherited_upper = inherited.upper()[[target_row]];
            if !candidate_lower.is_finite()
                || !candidate_upper.is_finite()
                || !inherited_lower.is_finite()
                || !inherited_upper.is_finite()
                || candidate_lower > candidate_upper
                || inherited_lower > inherited_upper
            {
                return Err(NyError::SoundnessRefusal(
                    "output-conditioned two-seed backward produced malformed endpoints".to_string(),
                ));
            }
            let best_lower = if candidate_lower > inherited_lower {
                candidate_lower
            } else {
                inherited_lower
            };
            let best_upper = if candidate_upper < inherited_upper {
                candidate_upper
            } else {
                inherited_upper
            };
            if best_lower > best_upper {
                return Err(NyError::SoundnessRefusal(
                    "output-conditioned intersection crossed its inherited interval".to_string(),
                ));
            }
            lower.push(best_lower);
            upper.push(best_upper);
        }
        Ok((lower, upper))
    }
}

/// Build a minimal placeholder `[n]`-shaped BoundedTensor used only to derive a
/// flat `GraphTargetShapeContract` (1D restore) for an objective chunk.
fn flat_bounds_view(n: usize) -> Result<BoundedTensor> {
    let lower = ArrayD::from_elem(ndarray::IxDyn(&[n]), 0.0_f32);
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[n]), 0.0_f32);
    BoundedTensor::new(lower, upper)
}

/// Build the seed `CrownBounds` for objective rows `r0..r1` (count
/// `chunk_rows = r1 - r0`) of a `target_dim`-row identity.
///
/// - When `spatial` is `Some((c, h, w))` and the range is the complete output
///   grid, returns the canonical full virtual Patches identity used by the
///   ordinary target backward.
/// - Otherwise returns a Dense `[chunk_rows x target_dim]` slice of the identity;
///   unlike a sparse Patches carrier, this has an unambiguous k-row contract.
fn build_chunk_seed(
    target_dim: usize,
    r0: usize,
    r1: usize,
    spatial: Option<(usize, usize, usize)>,
    deadline: Option<std::time::Instant>,
) -> Result<CrownBounds> {
    let rows: Vec<usize> = (r0..r1).collect();
    build_subset_seed(target_dim, &rows, spatial, deadline)
}

/// Build the seed `CrownBounds` for an ARBITRARY set of objective `rows` of a
/// `target_dim`-row identity. The contiguous chunk seed above is the special
/// case `rows = r0..r1`; the margin-subset backward (#margin-subset-seed)
/// passes the (sorted, possibly non-contiguous) spec-referenced rows.
///
/// - When `spatial` is `Some((c, h, w))` AND `rows` is the COMPLETE output grid
///   in flat order, returns the canonical full virtual Patches identity. This
///   is the same 6-D seed as the ordinary full target backward; using a
///   full-length `sparse_identity` here would make the first structural Conv2d
///   densify and then re-enter as 7-D Patches, changing certified reduction
///   order despite denoting the same mathematical identity.
/// - Otherwise returns a Dense `[rows.len() x target_dim]` selection of
///   identity rows.
///
/// # Why a STRICT subset never takes the sparse-identity Patches seed
///
/// `PatchesLinearBounds::sparse_identity` sets `row_count = rows.len()`, but
/// every conversion OUT of that layout expands it back to the full output grid:
/// `dense_rows_total` (patches/to_dense.rs) returns `out_dim` for a
/// non-explicit-rows sparse layout, and `sparse_identity_to_dense` writes one
/// scattered row per output POSITION (`lower_a[[flat, flat]] = 1.0`), leaving
/// the unselected positions as all-zero rows. That is the right contract for
/// the CROWN-IBP *partial* collector (`network/ibp/crown_partial.rs`), which
/// concretizes the full grid and then merges the zero rows back with IBP — but
/// it is the WRONG contract here: [`GraphNetwork::propagate_crown_to_node_subset`]
/// returns exactly `rows.len()` values, in `rows` order.
///
/// The two contracts are irreconcilable, and the disagreement is caught by
/// `PatchesLinearBounds::dense_pair_shape` (bounds/patches.rs), which REFUSES a
/// `row_count != out_dim` sparse identity. So a strict-subset patches seed
/// errors out at the first STRUCTURAL layer of the walk (Conv2d / Linear /
/// pooling / Flatten — the `sparse_structural_to_dense` densify in
/// `sequential/crown/patches_step.rs`), which for a spatial target is its own
/// layer: the walk starts AT the target because `ancestors()` is inclusive.
/// Measured on yolo_2023, every `#spec-influence-cone` seed died that way
/// (`Conv_20` 1,568 of 5,408 rows, `Conv_25` 576 of 10,816) before doing any
/// work, and the collector fell back to the full-width backward — so the cone
/// paid its setup cost and bought nothing. Relaxing the `dense_pair_shape`
/// guard instead is NOT a fix: it makes the k-row and grid-width contracts
/// collide for real, which trips the `lower_b len != lower_a.nrows()` invariant
/// in `core/graph/backward_helpers.rs`.
///
/// The Dense k-row seed below is exact (a single `1.0` per row, no relaxation),
/// honors the k-row contract, and does NOT forfeit the patches representation:
/// `try_dense_spatial_patches_reentry` lifts the k-row dense relation back into
/// 7-D Patches at the target's own Conv2d whenever the re-entry fits the CPU
/// dense budget — with `row_count = k` and `unstable_idx = None`, a layout on
/// which every conversion path agrees.
fn build_subset_seed(
    target_dim: usize,
    rows: &[usize],
    spatial: Option<(usize, usize, usize)>,
    deadline: Option<std::time::Instant>,
) -> Result<CrownBounds> {
    let num_rows = rows.len();
    // Only the complete, in-order grid is the canonical full virtual identity
    // (`row_count == out_dim`, row i at flat position i). Anything else takes
    // the Dense selection.
    let covers_full_grid_in_order =
        num_rows == target_dim && rows.iter().enumerate().all(|(i, &row)| row == i);
    if let Some((out_c, out_h, out_w)) = spatial.filter(|_| covers_full_grid_in_order) {
        let spatial_dim = out_c
            .checked_mul(out_h)
            .and_then(|value| value.checked_mul(out_w))
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "target input-linear full-grid spatial shape overflows usize".to_string(),
                )
            })?;
        if spatial_dim != target_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![target_dim],
                got: vec![spatial_dim],
            });
        }
        let shape = (out_c, out_h, out_w);
        Ok(CrownBounds::Patches(Box::new(
            PatchesLinearBounds::try_identity_with_deadline(shape, shape, deadline, 0)?,
        )))
    } else {
        // Budget guard for the rows REDIRECTED here from the patches branch: a
        // spatial target previously allocated nothing for its seed (virtual
        // sparse identity), so charge the `[k x target_dim]` pair against the
        // one CPU dense budget before allocating it. Refusing with the
        // structured `CpuMemoryExceeded` is fail-open — both call sites map ANY
        // error to "skip the subset, run the historical full-width path".
        //
        // Deliberately NOT applied when `spatial` is `None`: that branch always
        // allocated this pair, and gating it now would newly refuse seeds the
        // chunk driver already sizes itself (`auto_objective_chunk_rows` can
        // clamp to a single row whose pair still exceeds the budget). Leaving it
        // ungated keeps the non-spatial path byte-for-byte unchanged.
        if spatial.is_some() {
            let estimate =
                DenseMaterializationEstimate::new("margin_subset_dense_seed", num_rows, target_dim);
            let budget = cpu_crown_dense_budget_bytes();
            if estimate.exceeds_budget(budget) {
                return Err(NyError::CpuMemoryExceeded {
                    required_bytes: estimate.required_bytes,
                    budget_bytes: budget,
                    site: "margin_subset_dense_seed",
                });
            }
        }
        // Dense identity rows at the requested positions.
        let mut lower_a = Array2::<f32>::zeros((num_rows, target_dim));
        let mut upper_a = Array2::<f32>::zeros((num_rows, target_dim));
        for (i, &col) in rows.iter().enumerate() {
            lower_a[[i, col]] = 1.0;
            upper_a[[i, col]] = 1.0;
        }
        let lb = LinearBounds::new(
            lower_a,
            Array1::zeros(num_rows),
            upper_a,
            Array1::zeros(num_rows),
        )?;
        Ok(CrownBounds::Dense(lb))
    }
}

#[cfg(test)]
mod patches_densify_budget_tests {
    use super::{materialize_optional_resnet_seed, patches_densify_over_budget};
    use crate::bounds::patches::{CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds};
    use crate::bounds::LinearBounds;
    use ndarray::{Array1, ArrayD, IxDyn};
    use ny_core::NyError;

    /// #patches-row-range: the mid-walk densify guard refuses a Patches
    /// relation whose dense pair exceeds the budget with the structured
    /// `CpuMemoryExceeded` (mapped to a sound fallback by every CROWN caller),
    /// and passes under-budget relations through untouched.
    #[test]
    fn midwalk_densify_budget_guard() {
        // 1x2x2 identity: dense pair is 4x4x4x2 = 128 bytes — under budget.
        let small = PatchesLinearBounds::identity((1, 2, 2), (1, 2, 2));
        assert!(patches_densify_over_budget(&small, 1024).is_none());
        // Exactly at the budget is NOT over budget (strict >).
        let small_bytes = small.dense_pair_bytes().expect("estimable");
        assert!(patches_densify_over_budget(&small, small_bytes).is_none());
        assert!(patches_densify_over_budget(&small, small_bytes - 1).is_some());

        // VGG16 conv1 scale: 64x224x224 rows against a 3x224x224 input is a
        // ~3.9 TB dense pair — over any real budget, including the 2 GiB
        // default. (No dense allocation happens; identity patches are virtual.)
        let huge = PatchesLinearBounds::identity((64, 224, 224), (3, 224, 224));
        let err = patches_densify_over_budget(&huge, 2048 * 1024 * 1024)
            .expect("VGG-scale dense pair must trip the budget guard");
        assert!(err.is_cpu_memory_exceeded());
        // A large-enough budget lets it through (fits => None).
        assert!(patches_densify_over_budget(&huge, usize::MAX).is_none());
    }

    #[test]
    fn optional_resnet_seed_budget_refusal_borrows_anchored_patches() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_DENSE_BUDGET_MB", "0");

            let geometry = PatchGeometry::anchored(vec![0, 1], vec![0, 1])
                .expect("fixture axes are non-empty");
            let data = PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::from_elem(IxDyn(&[1, 2, 2, 1, 1, 1]), 0.5)),
                geometry,
                identity: false,
                output_shape: (1, 2, 2),
                input_shape: (1, 2, 2),
                unstable_idx: None,
            };
            let patches = PatchesLinearBounds {
                row_count: 4,
                lower_a: data.clone(),
                lower_b: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
                upper_a: data,
                upper_b: Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]),
            };
            let expected = patches.clone();
            let carrier = CrownBounds::Patches(Box::new(patches));

            let error = materialize_optional_resnet_seed(&carrier, None)
                .expect_err("zero budget must refuse optional resnet seed materialization");
            assert!(
                matches!(error, NyError::CpuMemoryExceeded { .. }),
                "expected typed memory refusal, got {error:?}"
            );
            let CrownBounds::Patches(actual) = &carrier else {
                panic!("borrowed optional seed changed the source carrier")
            };
            assert_eq!(actual.row_count, expected.row_count);
            assert_eq!(actual.lower_a.geometry, expected.lower_a.geometry);
            assert_eq!(actual.lower_a.patches, expected.lower_a.patches);
            assert_eq!(actual.lower_b, expected.lower_b);
            assert_eq!(actual.upper_a.geometry, expected.upper_a.geometry);
            assert_eq!(actual.upper_a.patches, expected.upper_a.patches);
            assert_eq!(actual.upper_b, expected.upper_b);
        });
    }

    #[test]
    fn optional_resnet_seed_expired_deadline_keeps_patches_unmaterialized() {
        let expected = PatchesLinearBounds::identity((1, 2, 2), (1, 2, 2));
        let carrier = CrownBounds::Patches(Box::new(expected.clone()));

        let error = materialize_optional_resnet_seed(&carrier, Some(std::time::Instant::now()))
            .expect_err("expired authority must refuse optional seed materialization");
        assert!(
            matches!(error, NyError::DeadlineExceeded(_)),
            "expected typed deadline refusal, got {error:?}"
        );
        let CrownBounds::Patches(actual) = &carrier else {
            panic!("borrowed optional seed changed the source carrier")
        };
        assert_eq!(actual.row_count, expected.row_count);
        assert_eq!(actual.lower_a.identity, expected.lower_a.identity);
        assert_eq!(actual.upper_a.identity, expected.upper_a.identity);
        assert_eq!(actual.lower_b, expected.lower_b);
        assert_eq!(actual.upper_b, expected.upper_b);
    }

    #[test]
    fn optional_resnet_dense_seed_is_borrowed_without_clone() {
        let carrier = CrownBounds::Dense(LinearBounds::identity(1));
        let expected = match &carrier {
            CrownBounds::Dense(bounds) => bounds,
            CrownBounds::Patches(_) => unreachable!(),
        };
        let materialized = materialize_optional_resnet_seed(&carrier, None)
            .expect("Dense optional seed should be borrowed");
        let std::borrow::Cow::Borrowed(actual) = materialized else {
            panic!("Dense optional seed must not clone before accelerator admission")
        };
        assert!(std::ptr::eq(actual, expected));
    }
}

#[cfg(test)]
mod margin_subset_backward_tests {
    use super::build_subset_seed;
    use crate::bounds::patches::CrownBounds;
    use crate::layers::{Conv2dLayer, Layer, LinearLayer, ReLULayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use crate::tests::with_crown_dense_budget_mb;
    use ndarray::{arr1, arr2, Array1, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;

    /// input(2) -> Linear(2->3) -> ReLU -> Linear(3->6): unstable ReLUs make
    /// the CROWN relaxation non-trivial, so row equality is a real check.
    fn dense_chain() -> (GraphNetwork, BoundedTensor) {
        let l1 = LinearLayer::new(
            arr2(&[[1.0_f32, -0.5], [0.25, 0.75], [-0.6, 0.4]]),
            Some(arr1(&[0.05_f32, -0.1, 0.02])),
        )
        .expect("l1");
        let l2 = LinearLayer::new(
            arr2(&[
                [0.9_f32, -0.3, 0.2],
                [-0.7, 0.6, -0.1],
                [0.4, 0.4, 0.4],
                [-0.2, -0.8, 0.5],
                [0.3, -0.6, -0.9],
                [0.8, 0.1, -0.4],
            ]),
            Some(arr1(&[0.01_f32, -0.02, 0.03, 0.0, -0.05, 0.04])),
        )
        .expect("l2");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("l1", Layer::Linear(l1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(l2),
            vec!["relu".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5]).into_dyn(),
            arr1(&[1.0_f32, 0.75]).into_dyn(),
        )
        .expect("input");
        (graph, input)
    }

    /// #margin-subset-seed: the subset backward's rows are BIT-IDENTICAL to
    /// the same rows of the full-width backward (dense identity seed path).
    #[test]
    fn subset_rows_match_full_backward_bit_identical_dense() {
        let (graph, input) = dense_chain();
        let forward = graph.collect_node_bounds(&input).expect("forward bounds");

        let full = graph
            .propagate_crown_to_node(
                &input,
                "out",
                &HashMap::new(),
                &forward,
                None,
                None,
                Some(0),
                None,
            )
            .expect("full-width backward");
        let full_flat = full.flatten();
        let rows = [1_usize, 4];
        let (lower, upper) = graph
            .propagate_crown_to_node_subset(
                &input,
                "out",
                &HashMap::new(),
                &forward,
                None,
                "margin-subset-test",
                None,
                false,
                false,
                &rows,
                None,
            )
            .expect("subset backward");
        for (i, &row) in rows.iter().enumerate() {
            assert_eq!(
                lower[i],
                full_flat.lower()[[row]],
                "lower row {row} must be bit-identical"
            );
            assert_eq!(
                upper[i],
                full_flat.upper()[[row]],
                "upper row {row} must be bit-identical"
            );
        }
        // Meaningfulness guard: the CROWN rows must actually tighten the IBP
        // rows somewhere among the chosen indices, or this test proves nothing.
        let ibp_out = forward.get("out").expect("out IBP").flatten();
        assert!(
            rows.iter().any(|&row| {
                full_flat.lower()[[row]] > ibp_out.lower()[[row]]
                    || full_flat.upper()[[row]] < ibp_out.upper()[[row]]
            }),
            "CROWN must beat IBP on at least one tested row"
        );
    }

    /// Spatial conv target via the collector (patches-override) walk: subset
    /// rows must match the full collector backward bit-for-bit as well.
    #[test]
    fn subset_rows_match_full_backward_bit_identical_conv_collector() {
        let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 2, 1, 1]), vec![0.9_f32, -0.35, -0.45, 0.8])
            .expect("kernel");
        let conv = Conv2dLayer::with_input_shape(
            kernel,
            Some(arr1(&[0.05_f32, -0.1])),
            (1, 1),
            (0, 0),
            2,
            2,
        )
        .expect("conv");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["conv".to_string()],
        ));
        graph.set_output("relu");
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[2, 2, 2]),
                vec![-1.0_f32, -0.6, 0.1, -0.3, -0.5, -0.2, 0.0, -0.4],
            )
            .expect("lower"),
            ArrayD::from_shape_vec(
                IxDyn(&[2, 2, 2]),
                vec![1.2_f32, 0.7, 0.9, 0.6, 0.8, 0.5, 1.0, 0.4],
            )
            .expect("upper"),
        )
        .expect("input");
        let forward = graph.collect_node_bounds(&input).expect("forward bounds");

        let full = graph
            .propagate_crown_to_node_for_collector(
                &input,
                "relu",
                &HashMap::new(),
                &forward,
                None,
                None,
                Some(0),
                None,
            )
            .expect("full collector backward");
        let full_flat = full.flatten();
        let rows = [0_usize, 3, 7];
        let (lower, upper) = graph
            .propagate_crown_to_node_subset(
                &input,
                "relu",
                &HashMap::new(),
                &forward,
                None,
                "margin-subset-test",
                None,
                false,
                true,
                &rows,
                None,
            )
            .expect("subset collector backward");
        for (i, &row) in rows.iter().enumerate() {
            assert_eq!(
                lower[i],
                full_flat.lower()[[row]],
                "lower row {row} must be bit-identical"
            );
            assert_eq!(
                upper[i],
                full_flat.upper()[[row]],
                "upper row {row} must be bit-identical"
            );
        }
    }

    /// #spec-influence-cone REGRESSION (seed representation).
    ///
    /// A STRICT subset of a spatial target must NOT be seeded as a
    /// sparse-identity Patches relation: `sparse_identity` sets
    /// `row_count = rows.len()` while every conversion out of that layout
    /// expands to the full `out_dim` grid, so `dense_pair_shape` refuses it and
    /// the walk dies at its first structural layer. The COMPLETE grid in flat
    /// order is the canonical full virtual Patches identity used by the
    /// ordinary full-width backward.
    #[test]
    fn strict_subset_seeds_dense_k_rows_full_grid_keeps_patches() {
        let spatial = (2_usize, 3_usize, 4_usize);
        let target_dim = 2 * 3 * 4;
        let rows = [0_usize, 5, 6, 23];

        let seed = build_subset_seed(target_dim, &rows, Some(spatial), None).expect("subset seed");
        let CrownBounds::Dense(lb) = &seed else {
            panic!("a strict subset must seed Dense: a sparse-identity Patches seed cannot honor the k-row contract");
        };
        assert_eq!(
            lb.num_outputs(),
            rows.len(),
            "seed must have exactly k rows"
        );
        assert_eq!(lb.num_inputs(), target_dim);
        for (i, &col) in rows.iter().enumerate() {
            for c in 0..target_dim {
                let want = if c == col { 1.0 } else { 0.0 };
                assert_eq!(lb.lower_a()[[i, c]], want, "row {i} col {c}");
                assert_eq!(lb.upper_a()[[i, c]], want, "row {i} col {c}");
            }
            assert_eq!(lb.lower_b()[i], 0.0);
            assert_eq!(lb.upper_b()[i], 0.0);
        }

        // Complete grid in flat order: use the canonical non-sparse virtual
        // identity, matching the ordinary full-width target backward exactly.
        let all: Vec<usize> = (0..target_dim).collect();
        let full =
            build_subset_seed(target_dim, &all, Some(spatial), None).expect("full-grid seed");
        let CrownBounds::Patches(pb) = &full else {
            panic!("the complete grid must keep a Patches seed");
        };
        assert_eq!(pb.row_count, target_dim);
        assert!(pb.lower_a.identity && pb.upper_a.identity);
        assert!(pb.lower_a.unstable_idx.is_none());
        assert!(pb.upper_a.unstable_idx.is_none());
        // Every conversion path must agree on this layout—which is exactly
        // what a strict-subset sparse identity fails to do.
        assert!(
            pb.dense_pair_shape().is_ok(),
            "full-grid virtual identity must have a derivable dense pair shape"
        );

        // A 1-D (non-spatial) target is unaffected: still the Dense selection.
        let flat = build_subset_seed(target_dim, &rows, None, None).expect("flat seed");
        assert!(matches!(flat, CrownBounds::Dense(_)));
    }

    /// input(8,8,8) -> Conv 1x1 -> ReLU -> Conv 1x1, all shapes (8,8,8).
    /// 512 flat outputs, so a 1 MiB dense budget makes the `[512 x 512]`
    /// identity pair (2 MiB) over budget and the target patches-eligible.
    fn spatial_conv_relu_conv() -> (GraphNetwork, BoundedTensor) {
        let kernel = |seed: f32| {
            let mut v = Vec::with_capacity(64);
            for o in 0..8 {
                for i in 0..8 {
                    let t = (o * 8 + i) as f32;
                    v.push((t * 0.37 + seed).sin() * 0.6);
                }
            }
            ArrayD::from_shape_vec(IxDyn(&[8, 8, 1, 1]), v).expect("kernel")
        };
        let bias = |seed: f32| {
            Some(Array1::from_shape_fn(8, |o| {
                (o as f32).mul_add(0.21, seed).cos() * 0.1
            }))
        };
        let conv1 = Conv2dLayer::with_input_shape(kernel(0.0), bias(0.0), (1, 1), (0, 0), 8, 8)
            .expect("conv1");
        let conv2 = Conv2dLayer::with_input_shape(kernel(1.3), bias(0.7), (1, 1), (0, 0), 8, 8)
            .expect("conv2");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["conv1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "conv2",
            Layer::Conv2d(conv2),
            vec!["relu".to_string()],
        ));
        graph.set_output("conv2");
        let lower = ArrayD::from_shape_fn(IxDyn(&[8, 8, 8]), |ix| {
            -0.5 - ((ix[0] + ix[1] + ix[2]) as f32) * 0.01
        });
        let upper = ArrayD::from_shape_fn(IxDyn(&[8, 8, 8]), |ix| {
            0.5 + ((ix[0] * 2 + ix[2]) as f32) * 0.01
        });
        let input = BoundedTensor::new(lower, upper).expect("input box");
        (graph, input)
    }

    /// #spec-influence-cone REGRESSION (end to end).
    ///
    /// A strict-subset backward on a SPATIAL, patches-eligible Conv2d target
    /// must complete and stay sound. Before the seed fix this returned
    /// `ShapeMismatch { expected: [512], got: [8] }` from `dense_pair_shape`
    /// at the target's OWN Conv2d — the exact shape of the yolo_2023 failure
    /// (`Conv_20` 1568/5408, `Conv_25` 576/10816) — and the collector silently
    /// fell back to the full-width backward, so the cone bought nothing.
    #[test]
    fn subset_backward_completes_on_spatial_patches_eligible_conv_target() {
        with_crown_dense_budget_mb("1", || {
            let (graph, input) = spatial_conv_relu_conv();
            let forward = graph.collect_node_bounds(&input).expect("forward bounds");
            let target_ibp = forward.get("conv2").expect("conv2 IBP");

            // Guard that this test actually exercises the patches-start path:
            // without it the seed would be Dense for an unrelated reason and
            // the regression would not be covered (the pre-existing conv
            // collector test falls into exactly that trap).
            assert!(
                graph.crown_ibp_target_can_start_in_patches_for_test("conv2", target_ibp),
                "test setup must make 'conv2' patches-eligible"
            );

            let rows = [0_usize, 1, 73, 200, 201, 340, 500, 511];
            let (lower, upper) = graph
                .propagate_crown_to_node_subset(
                    &input,
                    "conv2",
                    &HashMap::new(),
                    &forward,
                    None,
                    "spec-influence-cone-test",
                    None,
                    false,
                    true,
                    &rows,
                    None,
                )
                .expect("strict-subset backward on a spatial patches target must complete");
            assert_eq!(lower.len(), rows.len());
            assert_eq!(upper.len(), rows.len());

            // Soundness: evaluate the network at interior points of the box by
            // running the forward pass on degenerate (point) boxes — exact for
            // this Conv/ReLU chain — and require every returned row to enclose
            // the true value.
            for t in [0.0_f32, 0.5, 1.0] {
                let point = {
                    let l = input.lower();
                    let u = input.upper();
                    let v = ArrayD::from_shape_fn(l.raw_dim(), |ix| {
                        let lo = l[ix.clone()];
                        let hi = u[ix];
                        lo + (hi - lo) * t
                    });
                    BoundedTensor::new(v.clone(), v).expect("point box")
                };
                let exact = graph
                    .collect_node_bounds(&point)
                    .expect("point forward")
                    .get("conv2")
                    .expect("conv2 point")
                    .flatten();
                for (i, &row) in rows.iter().enumerate() {
                    let v = exact.lower()[[row]];
                    assert!(
                        lower[i] <= v + 1e-4 && upper[i] >= v - 1e-4,
                        "row {row}: subset bound [{}, {}] excludes the exact value {v}",
                        lower[i],
                        upper[i]
                    );
                }
            }

            // The subset rows must stay comparable to the same rows of the
            // full-width collector backward.
            //
            // NOT bit-identical, deliberately: the k-row seed is Dense (plus an
            // optional 7-D patches re-entry) while the full-width seed is a
            // VIRTUAL patches identity, so the two walks take different — both
            // sound — representations and accumulate float error differently.
            // Measured on this fixture the subset row is the slightly LOOSER of
            // the two (width 3.412 vs 3.318 on row 0, a 2.8% difference). The
            // 2x width bound catches a relaxation blow-up while tolerating the
            // representation difference; the point-enclosure checks above are
            // what pin soundness.
            // The strengthened materialization/concretization receipts keep the
            // 1 MiB cap authoritative. At 128 rows, the live input relation has
            // four 128x512 f32 coefficient/error matrices (1,048,576 bytes), two
            // f32 bias vectors (1,024 bytes), and f64/f32 endpoint work arrays
            // (3,072 bytes): 1,052,672 bytes total, just over the cap. A 127-row
            // chunk is 1,044,448 bytes and therefore fits. Row independence makes
            // the resulting five chunks the same full-target oracle.
            let full = graph
                .propagate_crown_to_node_for_collector(
                    &input,
                    "conv2",
                    &HashMap::new(),
                    &forward,
                    None,
                    None,
                    Some(127),
                    None,
                )
                .expect("full-width collector backward");
            let full_flat = full.flatten();
            for (i, &row) in rows.iter().enumerate() {
                let (fl, fu) = (full_flat.lower()[[row]], full_flat.upper()[[row]]);
                assert!(
                    lower[i].is_finite() && upper[i].is_finite() && lower[i] <= upper[i],
                    "row {row}: subset bound [{}, {}] must be a finite interval",
                    lower[i],
                    upper[i]
                );
                assert!(
                    lower[i] < fu && upper[i] > fl,
                    "row {row}: subset [{}, {}] does not overlap full-width [{fl}, {fu}]",
                    lower[i],
                    upper[i]
                );
                assert!(
                    (upper[i] - lower[i]) <= 2.0 * (fu - fl) + 1e-3,
                    "row {row}: subset width {} blew up against full-width width {}",
                    upper[i] - lower[i],
                    fu - fl
                );
            }
        });
    }

    /// Fail-closed request validation: empty and out-of-range row sets error
    /// (the consume site maps any error to the full-width fallback).
    #[test]
    fn subset_rejects_empty_and_out_of_range_rows() {
        let (graph, input) = dense_chain();
        let forward = graph.collect_node_bounds(&input).expect("forward bounds");
        assert!(graph
            .propagate_crown_to_node_subset(
                &input,
                "out",
                &HashMap::new(),
                &forward,
                None,
                "margin-subset-test",
                None,
                false,
                false,
                &[],
                None,
            )
            .is_err());
        assert!(graph
            .propagate_crown_to_node_subset(
                &input,
                "out",
                &HashMap::new(),
                &forward,
                None,
                "margin-subset-test",
                None,
                false,
                false,
                &[6],
                None,
            )
            .is_err());
    }
}

#[cfg(test)]
mod output_conditioned_two_seed_tests {
    use super::{
        build_output_conditioned_output_seed,
        fold_output_conditioned_additional_seed_error_for_cpu,
        output_conditioned_additional_seed_node_is_audited,
    };
    use crate::bounds::{patches::CrownBounds, GraphAlphaState};
    use crate::layers::{
        AddLayer, DivLayer, Layer, LinearLayer, MulBinaryLayer, ReLULayer, ReciprocalLayer,
        SigmoidLayer, SqrtLayer, TanhLayer, WhereLayer,
    };
    use crate::network::core::{GraphNetwork, GraphNode};
    use ndarray::{arr1, arr2};
    use ny_core::NyError;
    use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    fn scalar_linear_head() -> (GraphNetwork, BoundedTensor) {
        let identity = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("scalar identity Linear");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "target",
            Layer::Linear(identity.clone()),
        ));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Linear(identity),
            vec!["target".to_string()],
        ));
        graph.set_output("output");
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input box");
        (graph, input)
    }

    fn two_relu_head() -> (GraphNetwork, BoundedTensor) {
        let identity = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("scalar identity Linear");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("target", Layer::Linear(identity)));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["target".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu2",
            Layer::ReLU(ReLULayer),
            vec!["relu1".to_string()],
        ));
        graph.set_output("relu2");
        let input = BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input box");
        (graph, input)
    }

    fn relevant_alpha_relu_head() -> (GraphNetwork, BoundedTensor) {
        let identity =
            || LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("scalar identity Linear");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "pre_alpha",
            Layer::Linear(identity()),
        ));
        graph.add_node(GraphNode::new(
            "alpha_relu",
            Layer::ReLU(ReLULayer),
            vec!["pre_alpha".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "target",
            Layer::Linear(identity()),
            vec!["alpha_relu".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Linear(identity()),
            vec!["target".to_string()],
        ));
        graph.set_output("output");
        let input = BoundedTensor::new(arr1(&[-0.3_f32]).into_dyn(), arr1(&[0.7_f32]).into_dyn())
            .expect("input box");
        (graph, input)
    }

    fn populated_relevant_relu_alpha_state(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        ibp: &HashMap<String, BoundedTensor>,
    ) -> GraphAlphaState {
        let pre_activation = graph
            .relu_preactivation_bounds(
                "alpha_relu",
                input,
                ibp,
                "output-conditioned alpha-ReLU test",
            )
            .expect("alpha-ReLU pre-activation bounds");
        let mut alpha_state = GraphAlphaState::new();
        alpha_state
            .add_relu_node("alpha_relu", pre_activation, false)
            .expect("alpha-ReLU state");
        assert!(alpha_state.alpha("alpha_relu").is_some());
        alpha_state
    }

    fn crossing_relu_output_head() -> (GraphNetwork, BoundedTensor) {
        let identity = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("scalar identity Linear");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("target", Layer::Linear(identity)));
        graph.add_node(GraphNode::new(
            "output",
            Layer::ReLU(ReLULayer),
            vec!["target".to_string()],
        ));
        graph.set_output("output");
        let input = BoundedTensor::new(arr1(&[-0.3_f32]).into_dyn(), arr1(&[0.7_f32]).into_dyn())
            .expect("input box");
        (graph, input)
    }

    fn bypass_linear_head() -> (GraphNetwork, BoundedTensor) {
        let identity =
            || LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("scalar identity Linear");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("target", Layer::Linear(identity())));
        graph.add_node(GraphNode::new(
            "main",
            Layer::Linear(identity()),
            vec!["target".to_string()],
        ));
        graph.add_node(GraphNode::from_input("bypass", Layer::Linear(identity())));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Add(AddLayer),
            vec!["main".to_string(), "bypass".to_string()],
        ));
        graph.set_output("output");
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input box");
        (graph, input)
    }

    fn unaudited_mul_head() -> (GraphNetwork, BoundedTensor) {
        let identity =
            || LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("scalar identity Linear");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("target", Layer::Linear(identity())));
        graph.add_node(GraphNode::from_input("other", Layer::Linear(identity())));
        graph.add_node(GraphNode::binary(
            "output",
            Layer::MulBinary(MulBinaryLayer),
            "target",
            "other",
        ));
        graph.set_output("output");
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input box");
        (graph, input)
    }

    fn two_row_bypass_head() -> (GraphNetwork, BoundedTensor) {
        let identity = || {
            LinearLayer::new(arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]), None)
                .expect("2x2 identity Linear")
        };
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("target", Layer::Linear(identity())));
        graph.add_node(GraphNode::new(
            "main",
            Layer::Linear(identity()),
            vec!["target".to_string()],
        ));
        graph.add_node(GraphNode::from_input("bypass", Layer::Linear(identity())));
        graph.add_node(GraphNode::binary(
            "output",
            Layer::Add(AddLayer),
            "main",
            "bypass",
        ));
        graph.set_output("output");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .expect("input box");
        (graph, input)
    }

    fn evaluate(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        gamma_lower: f32,
        gamma_upper: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let ibp = graph.collect_node_bounds(input).expect("IBP");
        graph
            .propagate_output_conditioned_crown_to_node_subset(
                input,
                "target",
                &[0],
                objective,
                threshold,
                &[gamma_lower],
                &[gamma_upper],
                &HashMap::new(),
                &ibp,
                None,
                None,
                None,
            )
            .expect("output-conditioned two-seed evaluation")
    }

    fn assert_at_most_one_outward_ulp_lower(actual: f32, exact: f32) {
        assert!(
            (next_down_f32(exact)..=exact).contains(&actual),
            "lower endpoint {actual} is not an outward enclosure within one ulp of {exact}"
        );
    }

    fn assert_at_most_one_outward_ulp_upper(actual: f32, exact: f32) {
        assert!(
            (exact..=next_up_f32(exact)).contains(&actual),
            "upper endpoint {actual} is not an outward enclosure within one ulp of {exact}"
        );
    }

    /// The global target box is [-1, 1]. Under the active-row violation
    /// premise `output <= 0.25`, the upper two-seed expression with gamma=1 is
    /// exactly `target - output + 0.25 == 0.25`.
    #[test]
    fn two_seed_linear_premise_tightens_target_upper() {
        let (graph, input) = scalar_linear_head();
        let (lower, upper) = evaluate(&graph, &input, &[1.0], 0.25, 0.0, 1.0);
        assert_at_most_one_outward_ulp_lower(lower[0], -1.0);
        assert!(
            (0.25..=0.250_001).contains(&upper[0]),
            "directed rounding may widen the exact 0.25 oracle by a few ulps: {}",
            upper[0]
        );
        assert!(upper[0] < 1.0, "premise must tighten the global upper");
    }

    /// This exercises a genuine two-frontier merge through two ReLU nodes.
    /// The second ReLU is stable-active; the first crosses zero on [-0.5, 1].
    /// Its default lower slope is one, so under `relu(relu(x)) <= 0` the upper
    /// expression `x - relu(relu(x))` is exactly non-positive.
    #[test]
    fn two_relu_premise_tightens_crossing_preactivation_and_contains_witnesses() {
        let (graph, input) = two_relu_head();
        let (lower, upper) = evaluate(&graph, &input, &[1.0], 0.0, 0.0, 1.0);
        assert_at_most_one_outward_ulp_lower(lower[0], -0.5);
        assert!(
            (0.0..=2.0e-6).contains(&upper[0]),
            "two-frontier upper should enclose the exact zero boundary: {}",
            upper[0]
        );
        assert!(upper[0] < 1.0, "premise must tighten the global upper");
        for witness_target in [-0.5_f32, -0.25, 0.0] {
            assert!(
                lower[0] <= witness_target && witness_target <= upper[0],
                "premise-satisfying target {witness_target} escaped [{}, {}]",
                lower[0],
                upper[0]
            );
        }
    }

    /// The output frontier splits at Add: one contribution reaches `target`,
    /// while the bypass contribution reaches the network input independently.
    /// For output=2x and output<=0.2, gamma=0.5 makes
    /// `target - 0.5*(output-0.2) == 0.1`.
    #[test]
    fn bypass_frontier_is_preserved_until_input_accumulation() {
        let (graph, input) = bypass_linear_head();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let (lower, upper) = graph
            .propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                &[0],
                &[1.0],
                0.2,
                &[0.0],
                &[0.5],
                &HashMap::new(),
                &ibp,
                None,
                None,
                None,
            )
            .expect("bypass two-frontier evaluation");
        assert_at_most_one_outward_ulp_lower(lower[0], -1.0);
        assert!(
            (0.1..=0.100_002).contains(&upper[0]),
            "bypass contribution must survive to the final sum: {}",
            upper[0]
        );
        for witness_target in [-1.0_f32, -0.25, 0.1] {
            assert!(
                lower[0] <= witness_target && witness_target <= upper[0],
                "premise-satisfying bypass witness {witness_target} escaped"
            );
        }
    }

    /// A negative output coefficient exercises the opposite seed signs. The
    /// lower expression is
    /// `x + 0.25*(-2*x - 0.125) = 0.5*x - 0.03125`, whose exact box lower is
    /// -0.53125. The upper candidate is looser and must be discarded in favor
    /// of the inherited global upper 1.
    #[test]
    fn negative_objective_and_both_gamma_sides_match_exact_linear_oracle() {
        let (graph, input) = scalar_linear_head();
        let (lower, upper) = evaluate(&graph, &input, &[-2.0], 0.125, 0.25, 0.5);
        assert!(
            (-0.531_251..=-0.53125).contains(&lower[0]),
            "directed rounding may widen the exact -0.53125 oracle by a few ulps: {}",
            lower[0]
        );
        assert_at_most_one_outward_ulp_upper(upper[0], 1.0);
    }

    /// Zero gamma must reduce to the target identity path bit-for-bit.
    #[test]
    fn zero_gamma_is_bit_identical_to_unconditional_subset() {
        let (graph, input) = scalar_linear_head();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let baseline = graph
            .propagate_crown_to_node_subset(
                &input,
                "target",
                &HashMap::new(),
                &ibp,
                None,
                "zero-gamma baseline",
                None,
                false,
                false,
                &[0],
                None,
            )
            .expect("unconditional subset");
        let inherited = ibp.get("target").expect("target IBP").flatten();
        let expected_lower = baseline.0[0].max(inherited.lower()[[0]]);
        let expected_upper = baseline.1[0].min(inherited.upper()[[0]]);
        let conditioned = graph
            .propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                &[0],
                &[0.1],
                -0.2,
                &[0.0],
                &[0.0],
                &HashMap::new(),
                &ibp,
                None,
                None,
                None,
            )
            .expect("zero-gamma two-seed");
        assert_eq!(conditioned.0[0].to_bits(), expected_lower.to_bits());
        assert_eq!(conditioned.1[0].to_bits(), expected_upper.to_bits());
    }

    #[test]
    fn nonzero_gamma_refuses_relevant_alpha_relu_with_node_reason_before_execution() {
        let (graph, input) = relevant_alpha_relu_head();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let alpha_state = populated_relevant_relu_alpha_state(&graph, &input, &ibp);
        let error = graph
            .propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                &[0],
                &[0.1],
                0.0,
                &[0.7],
                &[0.6],
                &HashMap::new(),
                &ibp,
                Some(&alpha_state),
                None,
                None,
            )
            .expect_err("a populated relevant alpha-ReLU must fail admission");
        let NyError::SoundnessRefusal(reason) = error else {
            panic!("expected alpha-ReLU soundness refusal, got {error:?}");
        };
        assert!(
            reason.contains("alpha_relu"),
            "refusal must name the relevant node: {reason}"
        );
        assert!(
            reason.contains("alpha-ReLU"),
            "refusal must identify the unaudited alpha path: {reason}"
        );
    }

    #[test]
    fn zero_gamma_with_same_relevant_alpha_state_matches_ordinary_path_bits() {
        let (graph, input) = relevant_alpha_relu_head();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let crown = HashMap::new();
        let alpha_state = populated_relevant_relu_alpha_state(&graph, &input, &ibp);
        let ordinary = graph
            .propagate_crown_to_node_core(
                &input,
                "target",
                &crown,
                &ibp,
                Some(&alpha_state),
                None,
                "zero-gamma alpha-ReLU baseline",
                None,
                false,
                None,
                None,
                None,
            )
            .expect("ordinary alpha-ReLU target path")
            .flatten();
        let inherited = ibp.get("target").expect("target IBP").flatten();
        let expected_lower = ordinary.lower()[[0]].max(inherited.lower()[[0]]);
        let expected_upper = ordinary.upper()[[0]].min(inherited.upper()[[0]]);

        let conditioned = graph
            .propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                &[0],
                &[0.1],
                0.0,
                &[0.0],
                &[0.0],
                &crown,
                &ibp,
                Some(&alpha_state),
                None,
                None,
            )
            .expect("zero gamma must keep the ordinary populated-alpha path");
        assert_eq!(conditioned.0[0].to_bits(), expected_lower.to_bits());
        assert_eq!(conditioned.1[0].to_bits(), expected_upper.to_bits());
    }

    #[test]
    fn fixed_slope_non_dyadic_relu_gap_is_folded_and_encloses_f64_witnesses() {
        let (graph, input) = crossing_relu_output_head();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let objective = -0.1_f32;
        let threshold = -0.02_f32;
        let gamma_lower = 0.7_f32;

        let mut output_seed =
            build_output_conditioned_output_seed(&[objective], threshold, &[gamma_lower], &[0.0])
                .expect("non-dyadic output seed")
                .into_dense()
                .expect("dense output seed");
        assert!(
            output_seed.has_coeff_err(),
            "the non-dyadic gamma*objective seed product must carry its gap"
        );
        fold_output_conditioned_additional_seed_error_for_cpu(
            "output",
            &mut output_seed,
            &HashMap::new(),
            &ibp,
        )
        .expect("output seed gap must fold over the current output box");
        assert!(!output_seed.has_coeff_err());

        let target_box = ibp.get("target").expect("target pre-activation box");
        let mut irrelevant_alpha_state = GraphAlphaState::new();
        irrelevant_alpha_state
            .add_relu_node("irrelevant_relu", target_box, false)
            .expect("irrelevant alpha state");
        assert!(irrelevant_alpha_state.alpha("output").is_none());
        let mut through_relu = ReLULayer
            .propagate_linear_with_bounds(&output_seed, target_box)
            .expect("fixed-slope ReLU composition");
        let fresh_gap = through_relu
            .lower_a_err()
            .expect("ReLU must attach a fresh error channel")[[0, 0]];
        assert!(
            fresh_gap > 0.0,
            "the non-dyadic coefficient*chord product must have a certified fresh gap"
        );
        let lower_bias_before_fold = through_relu.lower_b()[0];
        fold_output_conditioned_additional_seed_error_for_cpu(
            "target",
            &mut through_relu,
            &HashMap::new(),
            &ibp,
        )
        .expect("fresh ReLU gap must fold over its pre-activation box");
        assert!(!through_relu.has_coeff_err());
        assert!(
            through_relu.lower_b()[0] < lower_bias_before_fold,
            "the positive fresh lower gap must widen the folded bias outward"
        );

        let (lower, upper) = graph
            .propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                &[0],
                &[objective],
                threshold,
                &[gamma_lower],
                &[0.0],
                &HashMap::new(),
                &ibp,
                Some(&irrelevant_alpha_state),
                None,
                None,
            )
            .expect("irrelevant alpha entry must retain fixed-slope ReLU evaluation");

        let dual_at =
            |x: f64| x + gamma_lower as f64 * (objective as f64 * x.max(0.0) - threshold as f64);
        let exact_f64_dual_lower = [-0.3_f32, 0.0, 0.7]
            .into_iter()
            .map(|x| dual_at(x as f64))
            .fold(f64::INFINITY, f64::min);
        assert!(
            lower[0] as f64 <= exact_f64_dual_lower,
            "folded lower rounded inward: got {}, exact f64 dual lower {}",
            lower[0],
            exact_f64_dual_lower
        );
        assert!(
            exact_f64_dual_lower - lower[0] as f64 <= 1.0e-4,
            "fresh-gap fold is unexpectedly loose: got {}, exact {}",
            lower[0],
            exact_f64_dual_lower
        );
        for witness in [0.25_f32, 0.4, 0.7] {
            let premise = objective as f64 * (witness as f64).max(0.0) - threshold as f64;
            assert!(premise <= 0.0, "chosen witness must satisfy the premise");
            assert!(
                lower[0] <= witness && witness <= upper[0],
                "f64 premise witness {witness} escaped [{}, {}]",
                lower[0],
                upper[0]
            );
        }
    }

    /// Seed products are stored with an explicit absolute error enclosure and
    /// biases round in their proof direction.
    #[test]
    fn seed_rounding_encloses_f64_product_and_bias_oracle() {
        let objective = [0.1_f32, -3.2_f32];
        let threshold = 0.3_f32;
        let gamma_lower = [0.7_f32];
        let gamma_upper = [0.6_f32];
        let seed =
            build_output_conditioned_output_seed(&objective, threshold, &gamma_lower, &gamma_upper)
                .expect("seed")
                .into_dense()
                .expect("dense seed");
        let lower_error = seed.lower_a_err().expect("lower error channel");
        let upper_error = seed.upper_a_err().expect("upper error channel");
        for col in 0..objective.len() {
            let exact_lower = gamma_lower[0] as f64 * objective[col] as f64;
            let exact_upper = -(gamma_upper[0] as f64) * objective[col] as f64;
            assert!(
                (seed.lower_a()[[0, col]] as f64 - exact_lower).abs()
                    <= lower_error[[0, col]] as f64
            );
            assert!(
                (seed.upper_a()[[0, col]] as f64 - exact_upper).abs()
                    <= upper_error[[0, col]] as f64
            );
        }
        let exact_lower_bias = -(gamma_lower[0] as f64) * threshold as f64;
        let exact_upper_bias = gamma_upper[0] as f64 * threshold as f64;
        assert!((seed.lower_b()[0] as f64) <= exact_lower_bias);
        assert!((seed.upper_b()[0] as f64) >= exact_upper_bias);
    }

    /// Exercise seed rounding, target/output merge, two affine backward steps,
    /// and final concretization against an exact-f64 one-dimensional oracle.
    #[test]
    fn full_linear_two_frontier_path_rounds_outward_against_f64_oracle() {
        let target_weight = 0.3_f32;
        let target_bias = 0.1_f32;
        let output_weight = -1.7_f32;
        let output_bias = 0.2_f32;
        let objective = 0.6_f32;
        let threshold = 0.05_f32;
        let gamma_lower = 0.98_f32;
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "target",
            Layer::Linear(
                LinearLayer::new(arr2(&[[target_weight]]), Some(arr1(&[target_bias])))
                    .expect("target Linear"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Linear(
                LinearLayer::new(arr2(&[[output_weight]]), Some(arr1(&[output_bias])))
                    .expect("output Linear"),
            ),
            vec!["target".to_string()],
        ));
        graph.set_output("output");
        let input_lower = -0.9_f32;
        let input_upper = 1.1_f32;
        let input = BoundedTensor::new(
            arr1(&[input_lower]).into_dyn(),
            arr1(&[input_upper]).into_dyn(),
        )
        .expect("input box");
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let (lower, upper) = graph
            .propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                &[0],
                &[objective],
                threshold,
                &[gamma_lower],
                &[0.0],
                &HashMap::new(),
                &ibp,
                None,
                None,
                None,
            )
            .expect("full two-frontier oracle evaluation");

        let target_at = |x: f64| target_weight as f64 * x + target_bias as f64;
        let output_at = |x: f64| output_weight as f64 * target_at(x) + output_bias as f64;
        let expression_at = |x: f64| {
            target_at(x) + gamma_lower as f64 * (objective as f64 * output_at(x) - threshold as f64)
        };
        let exact_expression_lower =
            expression_at(input_lower as f64).min(expression_at(input_upper as f64));
        let exact_target_lower = target_at(input_lower as f64).min(target_at(input_upper as f64));
        let exact_target_upper = target_at(input_lower as f64).max(target_at(input_upper as f64));
        let exact_intersection_lower = exact_expression_lower.max(exact_target_lower);

        assert!(
            lower[0] as f64 <= exact_intersection_lower,
            "lower rounded inward: got {}, exact best {}",
            lower[0],
            exact_intersection_lower
        );
        assert!(
            upper[0] as f64 >= exact_target_upper,
            "upper rounded inward: got {}, exact {}",
            upper[0],
            exact_target_upper
        );
        assert!(
            (lower[0] as f64) > exact_target_lower,
            "nonzero output premise should improve the inherited lower"
        );
        assert!(
            exact_intersection_lower - lower[0] as f64 <= 1.0e-4,
            "outward enclosure unexpectedly loose: got {}, exact {}",
            lower[0],
            exact_intersection_lower
        );
    }

    #[test]
    fn additional_seed_allowlist_explicitly_refuses_unaudited_special_branches() {
        for (layer, input_count) in [
            (Layer::MulBinary(MulBinaryLayer), 2),
            (Layer::Div(DivLayer), 2),
            (Layer::Where(WhereLayer::new()), 3),
            (Layer::Sigmoid(SigmoidLayer), 1),
            (Layer::Tanh(TanhLayer), 1),
            (Layer::Sqrt(SqrtLayer), 1),
            (Layer::Reciprocal(ReciprocalLayer), 1),
        ] {
            assert!(
                !output_conditioned_additional_seed_node_is_audited(&layer, input_count),
                "{} must stay outside the audited additional-seed subset",
                layer.layer_type()
            );
        }
        assert!(output_conditioned_additional_seed_node_is_audited(
            &Layer::ReLU(ReLULayer),
            1
        ));
        assert!(output_conditioned_additional_seed_node_is_audited(
            &Layer::Add(AddLayer),
            2
        ));
        assert!(!output_conditioned_additional_seed_node_is_audited(
            &Layer::Add(AddLayer),
            1
        ));
    }

    #[test]
    fn additional_seed_cpu_fold_uses_finite_ibp_fallback_and_clears_multirow_error() {
        let mut seed =
            build_output_conditioned_output_seed(&[0.1_f32, -3.2], 0.3, &[0.7, 0.6], &[0.6, 0.4])
                .expect("non-dyadic output seed")
                .into_dense()
                .expect("dense seed");
        let lower_error = seed.lower_a_err().expect("lower error channel").clone();
        let upper_error = seed.upper_a_err().expect("upper error channel").clone();
        assert!(
            lower_error
                .iter()
                .chain(upper_error.iter())
                .any(|e| *e > 0.0),
            "chosen products must exercise a nonzero seed error"
        );
        let old_lower_b = seed.lower_b().clone();
        let old_upper_b = seed.upper_b().clone();

        let invalid_crown = BoundedTensor::new_allow_infinite(
            arr1(&[f32::NEG_INFINITY, f32::NEG_INFINITY]).into_dyn(),
            arr1(&[f32::INFINITY, f32::INFINITY]).into_dyn(),
        )
        .expect("ordered but non-finite CROWN box");
        let finite_ibp = BoundedTensor::new(
            arr1(&[-2.0_f32, -1.0]).into_dyn(),
            arr1(&[3.0_f32, 4.0]).into_dyn(),
        )
        .expect("finite IBP box");
        let crown = HashMap::from([("node".to_string(), invalid_crown)]);
        let ibp = HashMap::from([("node".to_string(), finite_ibp)]);
        fold_output_conditioned_additional_seed_error_for_cpu("node", &mut seed, &crown, &ibp)
            .expect("finite IBP must safely discharge the seed error");
        assert!(!seed.has_coeff_err());

        let magnitudes = [3.0_f64, 4.0];
        for row in 0..2 {
            let lower_penalty: f64 = (0..2)
                .map(|col| lower_error[[row, col]] as f64 * magnitudes[col])
                .sum();
            let upper_penalty: f64 = (0..2)
                .map(|col| upper_error[[row, col]] as f64 * magnitudes[col])
                .sum();
            assert!(
                seed.lower_b()[row] as f64 <= old_lower_b[row] as f64 - lower_penalty,
                "lower row {row} did not fold outward"
            );
            assert!(
                seed.upper_b()[row] as f64 >= old_upper_b[row] as f64 + upper_penalty,
                "upper row {row} did not fold outward"
            );
        }
    }

    #[test]
    fn additional_seed_missing_current_node_box_refuses_instead_of_dropping_error() {
        let mut seed = build_output_conditioned_output_seed(&[0.1], 0.3, &[0.7], &[0.6])
            .expect("non-dyadic output seed")
            .into_dense()
            .expect("dense seed");
        assert!(seed.has_coeff_err());
        let error = fold_output_conditioned_additional_seed_error_for_cpu(
            "missing",
            &mut seed,
            &HashMap::new(),
            &HashMap::new(),
        )
        .expect_err("missing current-node box must fail closed");
        assert!(matches!(error, NyError::SoundnessRefusal(_)));
    }

    #[test]
    fn nonzero_gamma_refuses_mul_but_zero_gamma_keeps_unconditional_bits() {
        let (graph, input) = unaudited_mul_head();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let refused = graph
            .propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                &[0],
                &[0.1],
                0.0,
                &[0.7],
                &[0.6],
                &HashMap::new(),
                &ibp,
                None,
                None,
                None,
            )
            .expect_err("additional seed must reject MulBinary ancestry");
        assert!(matches!(refused, NyError::SoundnessRefusal(_)));

        let baseline = graph
            .propagate_crown_to_node_subset(
                &input,
                "target",
                &HashMap::new(),
                &ibp,
                None,
                "zero-gamma Mul ancestry baseline",
                None,
                false,
                false,
                &[0],
                None,
            )
            .expect("unconditional target subset");
        let inherited = ibp.get("target").expect("target IBP").flatten();
        let expected_lower = baseline.0[0].max(inherited.lower()[[0]]);
        let expected_upper = baseline.1[0].min(inherited.upper()[[0]]);
        let zero_gamma = graph
            .propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                &[0],
                &[0.1],
                0.0,
                &[0.0],
                &[0.0],
                &HashMap::new(),
                &ibp,
                None,
                None,
                None,
            )
            .expect("zero gamma must bypass additional-seed admission");
        assert_eq!(zero_gamma.0[0].to_bits(), expected_lower.to_bits());
        assert_eq!(zero_gamma.1[0].to_bits(), expected_upper.to_bits());
    }

    #[test]
    fn non_dyadic_multirow_dag_two_frontier_bounds_contain_premise_witnesses() {
        let (graph, input) = two_row_bypass_head();
        let objective = [0.1_f32, -0.2];
        let threshold = 0.1_f32;
        let gammas_lower = [0.7_f32, 0.6];
        let gammas_upper = [0.5_f32, 0.4];
        let seed = build_output_conditioned_output_seed(
            &objective,
            threshold,
            &gammas_lower,
            &gammas_upper,
        )
        .expect("multirow non-dyadic seed")
        .into_dense()
        .expect("dense seed");
        assert!(seed.has_coeff_err());

        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let (lower, upper) = graph
            .propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                &[0, 1],
                &objective,
                threshold,
                &gammas_lower,
                &gammas_upper,
                &HashMap::new(),
                &ibp,
                None,
                None,
                None,
            )
            .expect("audited multirow DAG two-frontier evaluation");
        assert_eq!(lower.len(), 2);
        assert_eq!(upper.len(), 2);
        assert!(lower
            .iter()
            .zip(&upper)
            .all(|(&lo, &up)| lo.is_finite() && up.is_finite() && lo <= up));

        for x0 in [-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
            for x1 in [-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
                let output = [2.0 * x0, 2.0 * x1];
                let premise = objective[0] * output[0] + objective[1] * output[1] - threshold;
                if premise <= 0.0 {
                    assert!(
                        lower[0] <= x0 && x0 <= upper[0] && lower[1] <= x1 && x1 <= upper[1],
                        "premise witness ({x0}, {x1}) escaped rows [{}, {}] x [{}, {}]",
                        lower[0],
                        upper[0],
                        lower[1],
                        upper[1]
                    );
                }
            }
        }
    }

    #[test]
    fn malformed_gamma_rows_and_expired_deadline_fail_closed() {
        let (graph, input) = scalar_linear_head();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let call = |rows: &[usize],
                    gammas_lower: &[f32],
                    gammas_upper: &[f32],
                    deadline: Option<Instant>| {
            graph.propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                rows,
                &[1.0],
                0.25,
                gammas_lower,
                gammas_upper,
                &HashMap::new(),
                &ibp,
                None,
                None,
                deadline,
            )
        };
        assert!(call(&[0], &[-0.1], &[0.0], None).is_err());
        assert!(call(&[0], &[f32::NAN], &[0.0], None).is_err());
        assert!(call(&[0], &[0.0], &[-0.1], None).is_err());
        assert!(call(&[0], &[0.0], &[f32::INFINITY], None).is_err());
        assert!(call(&[0], &[0.0], &[], None).is_err());
        assert!(call(&[], &[], &[], None).is_err());
        assert!(call(&[1], &[0.0], &[0.0], None).is_err());
        assert!(call(&[0, 0], &[0.0, 0.0], &[0.0, 0.0], None).is_err());
        assert!(call(&[0; 9], &[0.0; 9], &[0.0; 9], None).is_err());
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("monotonic subtraction");
        let error =
            call(&[0], &[0.0], &[0.0], Some(expired)).expect_err("expired deadline must fail");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));

        // Admission precedes row validation, topology lookup, and dense seed
        // construction: even a malformed request must take the same O(1)
        // typed-refusal route while its finite authority is still live.
        let live = Instant::now() + Duration::from_secs(5);
        let error = call(&[], &[], &[], Some(live))
            .expect_err("live finite output-conditioned requests must decline before setup");
        assert!(matches!(error, NyError::UnsupportedConfiguration(_)));
    }

    #[test]
    fn nonfinite_objective_or_threshold_is_refused_even_for_zero_gamma() {
        let (graph, input) = scalar_linear_head();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let call = |objective: &[f32], threshold: f32| {
            graph.propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                &[0],
                objective,
                threshold,
                &[0.0],
                &[0.0],
                &HashMap::new(),
                &ibp,
                None,
                None,
                None,
            )
        };
        assert!(call(&[f32::NAN], 0.0).is_err());
        assert!(call(&[f32::INFINITY], 0.0).is_err());
        assert!(call(&[1.0], f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn target_outside_output_ancestry_is_refused() {
        let (mut graph, input) = scalar_linear_head();
        let dead = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("dead scalar identity Linear");
        graph.add_node(GraphNode::from_input("dead_target", Layer::Linear(dead)));
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let error = graph
            .propagate_output_conditioned_crown_to_node_subset(
                &input,
                "dead_target",
                &[0],
                &[1.0],
                0.25,
                &[0.0],
                &[0.0],
                &HashMap::new(),
                &ibp,
                None,
                None,
                None,
            )
            .expect_err("dead target must be refused");
        assert!(matches!(error, NyError::InvalidSpec(_)));
    }

    #[test]
    fn output_seed_representation_is_dense() {
        let seed = build_output_conditioned_output_seed(&[1.0], 0.0, &[1.0], &[1.0]).expect("seed");
        assert!(matches!(seed, CrownBounds::Dense(_)));
    }
}

#[cfg(test)]
mod target_input_linear_tests {
    use super::{
        target_allows_patches_start, target_input_linear_chunk_row_bytes,
        target_input_linear_chunk_rows, target_input_linear_fixed_bytes, TargetInputLinearLimits,
    };
    use crate::layers::{
        AddLayer, Conv2dLayer, ConvTranspose2dLayer, Layer, LinearLayer, ReLULayer,
    };
    use crate::network::core::{GraphNetwork, GraphNode, NETWORK_INPUT};
    use crate::tests::with_serialized_env_vars;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    fn affine_relu() -> (GraphNetwork, BoundedTensor) {
        let affine = LinearLayer::new(
            arr2(&[[0.7_f32, -0.2], [-0.35, 0.8], [0.45, 0.55]]),
            Some(arr1(&[0.1_f32, -0.05, 0.02])),
        )
        .expect("affine");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("affine", Layer::Linear(affine)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["affine".to_string()],
        ));
        graph.set_output("relu");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.6]).into_dyn(),
            arr1(&[0.9_f32, 1.1]).into_dyn(),
        )
        .expect("input");
        (graph, input)
    }

    fn aggregate_plan(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        target: &str,
        crown: &HashMap<String, BoundedTensor>,
        ibp: &HashMap<String, BoundedTensor>,
    ) -> (usize, usize) {
        let input_dim = input.len();
        let target_dim = if target == NETWORK_INPUT {
            input_dim
        } else {
            ibp.get(target).expect("target bound").len()
        };
        let fixed =
            target_input_linear_fixed_bytes(target_dim, input_dim).expect("fixed workspace bytes");
        if target == NETWORK_INPUT {
            return (fixed, 0);
        }
        let relevant = graph.ancestors(target).expect("target ancestors");
        let mut relation_cols_sum = target_dim + input_dim;
        let mut relation_count = 2usize;
        for name in &relevant {
            relation_cols_sum += crown
                .get(name)
                .or_else(|| ibp.get(name))
                .expect("relation bound")
                .len();
            relation_count += 1;
        }
        let chunk_row =
            target_input_linear_chunk_row_bytes(target_dim, relation_cols_sum, relation_count)
                .expect("chunk-row bytes");
        (fixed, chunk_row)
    }

    fn assert_certificate_encloses_direct(
        certificate: &crate::bounds::LinearBounds,
        input: &BoundedTensor,
        direct: &BoundedTensor,
    ) {
        let from_certificate = certificate.concretize_sound(input);
        let direct = direct.flatten();
        assert_eq!(from_certificate.len(), direct.len());
        for row in 0..direct.len() {
            assert!(
                from_certificate.lower()[[row]] <= direct.lower()[[row]],
                "certificate lower row {row}={} exceeds direct CROWN lower={}",
                from_certificate.lower()[[row]],
                direct.lower()[[row]]
            );
            assert!(
                from_certificate.upper()[[row]] >= direct.upper()[[row]],
                "certificate upper row {row}={} is below direct CROWN upper={}",
                from_certificate.upper()[[row]],
                direct.upper()[[row]]
            );
        }
    }

    #[test]
    fn input_linear_forced_chunks_stitch_raw_error_then_publicly_fold_it() {
        let (graph, input) = affine_relu();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let direct = graph
            .propagate_crown_to_node(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                Some(0),
                None,
            )
            .expect("direct CROWN");
        let full = graph
            .propagate_crown_input_linear_to_node(&input, "relu", &HashMap::new(), &ibp, None, None)
            .expect("input-linear CROWN");
        assert_eq!((full.num_outputs(), full.num_inputs()), (3, 2));
        assert!(!full.has_coeff_err(), "public error must be discharged");

        let raw_full = graph
            .capture_crown_input_linear_to_node_raw_with_limits(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                TargetInputLinearLimits::PRODUCTION,
            )
            .expect("raw input-linear CROWN");
        assert!(
            raw_full.has_coeff_err(),
            "affine backward should produce a private certified error channel"
        );
        let flat_input = input.flatten();
        let mut expected_public = raw_full.clone();
        expected_public.fold_coeff_err_into_bias(
            flat_input.lower().as_slice().expect("lower slice"),
            flat_input.upper().as_slice().expect("upper slice"),
        );
        assert!(!expected_public.has_coeff_err());
        assert_eq!(full.lower_a(), expected_public.lower_a());
        assert_eq!(full.upper_a(), expected_public.upper_a());
        assert_eq!(full.lower_b(), expected_public.lower_b());
        assert_eq!(full.upper_b(), expected_public.upper_b());

        // Set the aggregate cap to exactly assembly + one conservative chunk
        // row. This deterministically forces one identity row per pass.
        let (fixed, chunk_row) = aggregate_plan(&graph, &input, "relu", &HashMap::new(), &ibp);
        let one_row_cap = fixed + chunk_row;
        assert_eq!(
            target_input_linear_chunk_rows(3, fixed, chunk_row, false, false, one_row_cap)
                .expect("one-row plan"),
            1
        );
        let raw_chunked = graph
            .capture_crown_input_linear_to_node_raw_with_limits(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                TargetInputLinearLimits {
                    aggregate_bytes: one_row_cap,
                },
            )
            .expect("one-row raw input-linear CROWN");
        assert_eq!(raw_chunked.lower_a(), raw_full.lower_a());
        assert_eq!(raw_chunked.upper_a(), raw_full.upper_a());
        assert_eq!(raw_chunked.lower_b(), raw_full.lower_b());
        assert_eq!(raw_chunked.upper_b(), raw_full.upper_b());
        assert_eq!(raw_chunked.lower_a_err(), raw_full.lower_a_err());
        assert_eq!(raw_chunked.upper_a_err(), raw_full.upper_a_err());

        let chunked = graph
            .propagate_crown_input_linear_to_node_with_limits(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                TargetInputLinearLimits {
                    aggregate_bytes: one_row_cap,
                },
            )
            .expect("one-row input-linear CROWN");
        assert_eq!(chunked.lower_a(), full.lower_a());
        assert_eq!(chunked.upper_a(), full.upper_a());
        assert_eq!(chunked.lower_b(), full.lower_b());
        assert_eq!(chunked.upper_b(), full.upper_b());
        assert!(!chunked.has_coeff_err());
        assert_certificate_encloses_direct(&full, &input, &direct);
    }

    #[test]
    fn input_linear_preflight_rejects_fixed_and_one_row_aggregate_peaks() {
        let (graph, input) = affine_relu();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let (fixed, chunk_row) = aggregate_plan(&graph, &input, "relu", &HashMap::new(), &ibp);

        let fixed_err = graph
            .propagate_crown_input_linear_to_node_with_limits(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                TargetInputLinearLimits {
                    aggregate_bytes: fixed - 1,
                },
            )
            .expect_err("fixed assembly must not exceed the aggregate cap");
        assert!(matches!(
            fixed_err,
            NyError::CpuMemoryExceeded {
                site: "graph-alpha target input-linear aggregate workspace",
                ..
            }
        ));

        let row_err = graph
            .propagate_crown_input_linear_to_node_with_limits(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                TargetInputLinearLimits {
                    aggregate_bytes: fixed + chunk_row - 1,
                },
            )
            .expect_err("assembly plus one chunk row must not exceed the aggregate cap");
        assert!(matches!(
            row_err,
            NyError::CpuMemoryExceeded {
                site: "graph-alpha target input-linear aggregate workspace",
                ..
            }
        ));
    }

    #[test]
    fn input_linear_rejects_unknown_target_even_with_spoofed_bounds() {
        let (graph, input) = affine_relu();
        let mut ibp = graph.collect_node_bounds(&input).expect("IBP");
        ibp.insert(
            "ghost".to_string(),
            ibp.get("relu").expect("relu bound").clone(),
        );
        let err = graph
            .propagate_crown_input_linear_to_node(
                &input,
                "ghost",
                &HashMap::new(),
                &ibp,
                None,
                None,
            )
            .expect_err("unknown target must fail closed");
        assert!(
            matches!(err, NyError::InvalidSpec(message) if message.contains("not a graph node"))
        );
    }

    #[test]
    fn input_linear_accepts_only_the_network_input_sentinel_without_a_graph_node() {
        let (graph, input) = affine_relu();
        let certificate = graph
            .propagate_crown_input_linear_to_node(
                &input,
                NETWORK_INPUT,
                &HashMap::new(),
                &HashMap::new(),
                None,
                None,
            )
            .expect("network-input identity");
        assert_eq!(
            (certificate.num_outputs(), certificate.num_inputs()),
            (2, 2)
        );
        assert!(!certificate.has_coeff_err());
        assert_certificate_encloses_direct(&certificate, &input, &input);
    }

    #[test]
    fn input_linear_captures_dag_merge_and_input_skip() {
        let first = LinearLayer::new(
            arr2(&[[0.8_f32, -0.3], [0.4, 0.9]]),
            Some(arr1(&[0.1_f32, -0.05])),
        )
        .expect("first affine");
        let second = LinearLayer::new(
            arr2(&[[0.6_f32, -0.2], [-0.4, 0.7]]),
            Some(arr1(&[0.0_f32, 0.0])),
        )
        .expect("second affine");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("first", Layer::Linear(first)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["first".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "second",
            Layer::Linear(second),
            vec!["relu".to_string()],
        ));
        graph.add_node(GraphNode::binary(
            "residual",
            Layer::Add(AddLayer),
            NETWORK_INPUT,
            "second",
        ));
        graph.set_output("residual");
        let input = BoundedTensor::new(
            arr1(&[-0.5_f32, -0.5]).into_dyn(),
            arr1(&[0.5_f32, 0.5]).into_dyn(),
        )
        .expect("input");
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let direct = graph
            .propagate_crown_to_node(
                &input,
                "residual",
                &HashMap::new(),
                &ibp,
                None,
                None,
                Some(0),
                None,
            )
            .expect("direct DAG CROWN");
        let certificate = graph
            .propagate_crown_input_linear_to_node(
                &input,
                "residual",
                &HashMap::new(),
                &ibp,
                None,
                None,
            )
            .expect("DAG input-linear CROWN");
        assert_eq!(
            (certificate.num_outputs(), certificate.num_inputs()),
            (2, 2)
        );
        assert!(!certificate.has_coeff_err());
        assert_certificate_encloses_direct(&certificate, &input, &direct);
    }

    #[test]
    fn input_linear_captures_patches_conv_chain() {
        let in_c = 4usize;
        let in_h = 20usize;
        let in_w = 25usize;
        let input_dim = in_c * in_h * in_w;
        let conv1 = Conv2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[2, in_c, 2, 2]), 0.125_f32),
            Some(arr1(&[0.03_f32, -0.02])),
            (2, 2),
            (0, 0),
            in_h,
            in_w,
        )
        .expect("conv1");
        let conv2 = Conv2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[1, 2, 3, 3]), -0.075_f32),
            Some(arr1(&[0.01_f32])),
            (1, 1),
            (0, 0),
            10,
            12,
        )
        .expect("conv2");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["conv1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "conv2",
            Layer::Conv2d(conv2),
            vec!["relu".to_string()],
        ));
        graph.set_output("conv2");
        graph.set_use_patches_mode(true);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[in_c, in_h, in_w]), -0.5_f32),
            ArrayD::from_elem(IxDyn(&[in_c, in_h, in_w]), 0.5_f32),
        )
        .expect("input");
        assert_eq!(input.len(), input_dim);
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let target = ibp.get("conv2").expect("target bound");
        let relevant = graph.ancestors("conv2").expect("ancestors");

        // The final exact Patches -> Dense capture now receipts both output
        // matrices and its live unfold plan (5,160,008 bytes for this fixture).
        // Eight MiB admits that certified peak with bounded allocator headroom.
        // Select the independent cost-admission route explicitly: a memory-
        // admitted Patches seed necessarily has a dense pair larger than the
        // same budget and therefore cannot also test successful terminal Dense
        // capture. Divisor 4096 sets the fixed 2 GiB calibration threshold to
        // 512 KiB, below this fixture's 1.28 MiB backward pair. The separate
        // zero-budget tests cover transactional refusal.
        let (certificate, raw_certificate, direct) = with_serialized_env_vars(
            &[
                ("NY_DENSE_BUDGET_MB", "8"),
                ("NY_PATCHES_COST_DIVISOR", "4096"),
            ],
            || {
                assert!(
                    target_allows_patches_start(&graph, "conv2", None, &relevant, target, false,),
                    "fixture must start capture in Patches form"
                );
                let certificate = graph
                    .propagate_crown_input_linear_to_node(
                        &input,
                        "conv2",
                        &HashMap::new(),
                        &ibp,
                        None,
                        None,
                    )
                    .expect("Patches input-linear CROWN");
                let raw_certificate = graph
                    .capture_crown_input_linear_to_node_raw_with_limits(
                        &input,
                        "conv2",
                        &HashMap::new(),
                        &ibp,
                        None,
                        None,
                        TargetInputLinearLimits::PRODUCTION,
                    )
                    .expect("raw Patches input-linear CROWN");
                let direct = graph
                    .propagate_crown_to_node(
                        &input,
                        "conv2",
                        &HashMap::new(),
                        &ibp,
                        None,
                        None,
                        Some(0),
                        None,
                    )
                    .expect("direct Patches CROWN");
                (certificate, raw_certificate, direct)
            },
        );
        assert_eq!(
            (certificate.num_outputs(), certificate.num_inputs()),
            (80, input_dim)
        );
        assert!(!certificate.has_coeff_err());
        assert!(
            raw_certificate.has_coeff_err(),
            "raw Conv capture must retain its certified coefficient error"
        );

        // Both APIs now start from the same canonical full virtual identity.
        // Before the fix, input-linear capture used a full-length sparse
        // identity, which forced Dense Conv2d followed by 7-D Patches re-entry;
        // direct CROWN stayed on native 6-D Patches. Their alternative sound
        // reductions crossed by one ULP, so direct CROWN was not evidence of a
        // missing error charge in the public certificate. With identical seed
        // and walk, the raw error-carrying certificate is bit-identical to the
        // direct result before public error folding.
        let raw_concrete = raw_certificate.concretize_sound(&input);
        let direct_flat = direct.flatten();
        for row in 0..direct_flat.len() {
            assert_eq!(
                raw_concrete.lower()[[row]].to_bits(),
                direct_flat.lower()[[row]].to_bits(),
                "raw/direct lower parity at row {row}"
            );
            assert_eq!(
                raw_concrete.upper()[[row]].to_bits(),
                direct_flat.upper()[[row]].to_bits(),
                "raw/direct upper parity at row {row}"
            );
        }
        assert_certificate_encloses_direct(&certificate, &input, &direct);

        // Independent exact extremum oracle. Conv1's k2/s2 windows contain
        // 4 channels x 2 x 2 = 16 positive 0.125 coefficients, so their shared
        // affine sum ranges exactly over [-1, 1]. ReLU is monotone and every
        // Conv2 coefficient is negative: all -0.5 inputs therefore realize the
        // output maximum (both ReLUs are zero), while all +0.5 inputs realize
        // the output minimum. Every valid k3 Conv2 output contains nine such
        // spatial pairs. Step the computed lower endpoint down in binary64 to
        // make the independently evaluated arithmetic a strict lower floor.
        let conv1_sum_max = 16.0_f64 * f64::from(0.125_f32) * f64::from(0.5_f32);
        assert_eq!(conv1_sum_max, 1.0);
        let relu_pair_max = (conv1_sum_max + f64::from(0.03_f32)).max(0.0)
            + (conv1_sum_max + f64::from(-0.02_f32)).max(0.0);
        let realized_lower = f64::from(0.01_f32) + 9.0_f64 * f64::from(-0.075_f32) * relu_pair_max;
        let realized_lower_floor =
            (0..8).fold(realized_lower, |value, _| ny_core::dd::next_down_f64(value));
        let realized_upper = f64::from(0.01_f32);
        let certificate_concrete = certificate.concretize_sound(&input);
        for row in 0..certificate_concrete.len() {
            assert!(
                f64::from(certificate_concrete.lower()[[row]]) <= realized_lower_floor,
                "certificate lower row {row} does not enclose exact monotone minimum"
            );
            assert!(
                f64::from(certificate_concrete.upper()[[row]]) >= realized_upper,
                "certificate upper row {row} does not enclose exact monotone maximum"
            );
        }
    }

    #[test]
    fn input_linear_no_deadline_matches_direct_and_finite_refuses() {
        let conv = ConvTranspose2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), -0.75_f32),
            Some(arr1(&[0.125_f32])),
            (1, 1),
            (0, 0),
            1,
            33,
        )
        .expect("ConvTranspose2d");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("convt", Layer::ConvTranspose2d(conv)));
        graph.set_output("convt");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 33]), -0.4_f32),
            ArrayD::from_elem(IxDyn(&[1, 1, 33]), 0.6_f32),
        )
        .expect("input");
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let direct = graph
            .propagate_crown_to_node(
                &input,
                "convt",
                &HashMap::new(),
                &ibp,
                None,
                None,
                Some(0),
                None,
            )
            .expect("direct ConvTranspose CROWN");
        let certificate = graph
            .propagate_crown_input_linear_to_node(
                &input,
                "convt",
                &HashMap::new(),
                &ibp,
                None,
                None,
            )
            .expect("no-deadline ConvTranspose input-linear CROWN");
        assert_eq!(
            (certificate.num_outputs(), certificate.num_inputs()),
            (33, 33)
        );
        assert!(!certificate.has_coeff_err());
        assert_certificate_encloses_direct(&certificate, &input, &direct);

        let error = graph
            .propagate_crown_input_linear_to_node(
                &input,
                "convt",
                &HashMap::new(),
                &ibp,
                None,
                Some(Instant::now() + Duration::from_secs(10)),
            )
            .expect_err("live finite input-linear capture must decline before allocation");
        assert!(
            matches!(error, NyError::UnsupportedConfiguration(_)),
            "expected typed finite-capture refusal, got {error:?}"
        );
    }

    #[test]
    fn input_linear_rejects_expired_deadline_before_allocation() {
        let (graph, input) = affine_relu();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("monotonic subtraction");
        let err = graph
            .propagate_crown_input_linear_to_node(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                Some(expired),
            )
            .expect_err("expired deadline must fail");
        assert!(matches!(err, NyError::DeadlineExceeded(_)));
    }
}

#[cfg(test)]
mod cut_segment_env_tests {
    use super::{
        effective_target_chunk_size, enforce_expected_fixed_wave_plan,
        objective_chunk_driver_route, objective_chunk_route_plan, parse_crown_cut_segment,
        ObjectiveChunkDriverRoute, PartialCrownDeadlineSalvagePolicy, TestChunkDeadlineGuard,
        DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS,
    };
    use crate::layers::{ConvTranspose2dLayer, Layer, ReLULayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
    use ndarray::{arr1, ArrayD, IxDyn};
    use ny_core::{GemmEngine, NaiveCpuGemmEngine, NyError, Result};
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    #[derive(Default)]
    struct CountingGemmEngine {
        calls: AtomicUsize,
    }

    impl GemmEngine for CountingGemmEngine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }
    }

    fn direct_conv_transpose_33() -> (GraphNetwork, BoundedTensor) {
        let conv = ConvTranspose2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), -0.75_f32),
            Some(arr1(&[0.125_f32])),
            (1, 1),
            (0, 0),
            1,
            33,
        )
        .expect("1x1 ConvTranspose2d");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("convt", Layer::ConvTranspose2d(conv)));
        graph.set_output("convt");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 33]), -0.4_f32),
            ArrayD::from_elem(IxDyn(&[1, 1, 33]), 0.6_f32),
        )
        .expect("input box");
        (graph, input)
    }

    fn demanded_conv_transpose_33() -> (GraphNetwork, BoundedTensor) {
        let (mut graph, input) = direct_conv_transpose_33();
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["convt".to_string()],
        ));
        graph.set_output("relu");
        (graph, input)
    }

    /// #crown-cut-segment gate parsing: unset/empty/non-numeric/0 all mean
    /// DISABLED (byte-identical full-prefix backward); only a positive integer
    /// enables cuts.
    #[test]
    fn parse_crown_cut_segment_defaults_off() {
        assert_eq!(parse_crown_cut_segment(None), 0);
        assert_eq!(parse_crown_cut_segment(Some("")), 0);
        assert_eq!(parse_crown_cut_segment(Some("0")), 0);
        assert_eq!(parse_crown_cut_segment(Some("abc")), 0);
        assert_eq!(parse_crown_cut_segment(Some("-4")), 0);
        assert_eq!(parse_crown_cut_segment(Some("1.5")), 0);
    }

    #[test]
    fn parse_crown_cut_segment_accepts_positive() {
        assert_eq!(parse_crown_cut_segment(Some("1")), 1);
        assert_eq!(parse_crown_cut_segment(Some("4")), 4);
        assert_eq!(parse_crown_cut_segment(Some(" 8 ")), 8);
    }

    #[test]
    fn deadline_conv_transpose_chunk_cap_preserves_no_deadline_policy() {
        let cap = DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS;
        assert_eq!(effective_target_chunk_size(0, false, true), 0);
        assert_eq!(effective_target_chunk_size(64, false, true), 64);
        assert_eq!(effective_target_chunk_size(0, true, false), 0);
        assert_eq!(effective_target_chunk_size(0, true, true), cap);
        assert_eq!(effective_target_chunk_size(cap * 2, true, true), cap);
        assert_eq!(effective_target_chunk_size(7, true, true), 7);
    }

    #[test]
    fn deadline_conv_transpose_route_retains_requested_and_effective_widths() {
        let plan = objective_chunk_route_plan(9_320, true, true);
        assert_eq!(plan.requested_rows, 9_320);
        assert_eq!(
            plan.effective_initial_rows,
            DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS
        );

        let no_deadline = objective_chunk_route_plan(9_320, false, true);
        assert_eq!(no_deadline.requested_rows, 9_320);
        assert_eq!(no_deadline.effective_initial_rows, 9_320);
    }

    #[test]
    fn objective_chunk_driver_models_fixed_wave_grouping_and_growth_fallbacks() {
        let fixed =
            objective_chunk_driver_route(28_800, 32, 9_320, true, false, true, false, true, 32);
        let ObjectiveChunkDriverRoute::FixedWaves(plan) = fixed else {
            panic!("deadline/no-cut/default-wave route must be fixed waves");
        };
        assert_eq!(plan.chunk_rows, 32);
        assert_eq!(plan.chunk_count, 900);
        assert_eq!(plan.wave_size, 32);
        assert_eq!(plan.wave_count, 29);

        assert_eq!(
            objective_chunk_driver_route(28_800, 32, 9_320, true, false, false, false, true, 32,),
            ObjectiveChunkDriverRoute::Sequential {
                adaptive_growth: true,
            },
            "wave kill switch must route to the adaptive sequential driver"
        );
        assert_eq!(
            objective_chunk_driver_route(28_800, 32, 9_320, true, true, true, false, false, 32,),
            ObjectiveChunkDriverRoute::Sequential {
                adaptive_growth: false,
            },
            "a cut context must route sequentially and honor the growth switch"
        );

        let memory_capped =
            objective_chunk_driver_route(28_800, 32, 64, true, false, true, false, true, 32);
        let ObjectiveChunkDriverRoute::FixedWaves(memory_capped) = memory_capped else {
            panic!("memory-capped route must remain fixed waves");
        };
        assert_eq!(memory_capped.wave_size, 2);
        assert_eq!(memory_capped.wave_count, 450);
    }

    #[test]
    fn fixed_wave_admission_is_bound_to_the_execution_route() {
        let fixed =
            objective_chunk_driver_route(28_800, 32, 9_320, true, false, true, false, true, 32);
        let ObjectiveChunkDriverRoute::FixedWaves(expected) = fixed else {
            panic!("fixture must select fixed waves");
        };
        assert_eq!(
            enforce_expected_fixed_wave_plan(fixed, Some(expected), "test", "target")
                .expect("identical runtime route"),
            fixed
        );

        let drifted =
            objective_chunk_driver_route(28_800, 32, 9_320, true, false, false, false, true, 32);
        assert_eq!(
            enforce_expected_fixed_wave_plan(drifted, None, "test", "target")
                .expect("dark callers carry no route expectation"),
            drifted,
            "no M1 expectation must preserve the live driver route"
        );
        let err = enforce_expected_fixed_wave_plan(drifted, Some(expected), "test", "target")
            .expect_err("wave kill-switch drift must fail before execution");
        let NyError::UnsupportedConfiguration(message) = err else {
            panic!("route drift must use the sound unsupported-route fallback");
        };
        assert!(message.contains("changed after M1 admission"));
        assert!(message.contains("target"));
    }

    /// A 33-row direct ConvTranspose target takes one coefficient GEMM pair
    /// without a deadline. Finite authority would select the 32-row objective
    /// chunk driver, whose whole-target setup is not yet cooperative, so it must
    /// refuse before either the injected engine or any chunk kernel runs.
    ///
    /// Serialized on the shared env lock: pin the ConvTranspose dead-work skip
    /// kill-switch off so the no-deadline baseline still exercises its
    /// historical engine pair; finite authority remains engine-free.
    #[test]
    fn deadline_conv_transpose_chunk_route_refuses_before_engine_work() {
        crate::tests::with_serialized_env_vars(&[("NY_CONV_SKIP_DEAD_F32", "0")], || {
            deadline_conv_transpose_chunk_route_refuses_before_engine_work_body();
        });
    }

    fn deadline_conv_transpose_chunk_route_refuses_before_engine_work_body() {
        let (graph, input) = direct_conv_transpose_33();
        let forward = graph.collect_node_bounds(&input).expect("forward bounds");

        let flat_engine = CountingGemmEngine::default();
        graph
            .propagate_crown_to_node(
                &input,
                "convt",
                &HashMap::new(),
                &forward,
                Some(&flat_engine),
                None,
                Some(0),
                None,
            )
            .expect("no-deadline flat ConvTranspose backward");
        assert_eq!(flat_engine.calls.load(Ordering::Relaxed), 2);

        let chunk_engine = CountingGemmEngine::default();
        let error = graph
            .propagate_crown_to_node(
                &input,
                "convt",
                &HashMap::new(),
                &forward,
                Some(&chunk_engine),
                Some(Instant::now() + Duration::from_secs(30)),
                Some(0),
                None,
            )
            .expect_err("finite objective-chunk setup must decline before execution");
        assert!(
            matches!(error, NyError::UnsupportedConfiguration(_)),
            "expected typed unsupported finite-chunk route, got {error:?}"
        );
        assert_eq!(
            chunk_engine.calls.load(Ordering::Relaxed),
            0,
            "finite objective-chunk refusal must precede the opaque engine"
        );
    }

    /// Controller-level regression for legacy no-deadline partial-row salvage.
    /// The test-only boundary injects a typed deadline before the second 32-row
    /// chunk without granting the driver a finite authority. This keeps the
    /// historical chunk/salvage mechanics covered while finite chunk admission
    /// itself remains fail-closed at O(1).
    #[test]
    fn deterministic_controller_deadline_retains_rows_on_legacy_chunk_route() {
        ny_test_utils::env::with_env_edits(|env| {
            for key in [
                "NY_CROWN_OBJ_CHUNK",
                "NY_NO_CHUNK_WAVE_PAR",
                "NY_NO_CHUNK_ABORT",
                "NY_NO_CHUNK_GROW",
                "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
                "NY_PATCHES_BUDGET_SECS",
            ] {
                env.remove(key);
            }
            env.set("NY_CROWN_OBJ_CHUNK", "32");

            let (graph, input) = demanded_conv_transpose_33();
            let ibp = graph.collect_node_bounds(&input).expect("certified IBP");
            let _injected = TestChunkDeadlineGuard::after_committed_chunks(1);
            let result = graph
                .collect_crown_ibp_bounds_core_inner_with_cut_segment_and_policies(
                    &input,
                    ibp.clone(),
                    None,
                    None,
                    None,
                    0,
                    false,
                    PartialCrownDeadlineSalvagePolicy::EnabledByExactEnvironment,
                    super::crown_tighten::CrownIbpCollectionMode::Standard,
                )
                .expect("collector must return its sound partial-row map");

            assert!(matches!(
                result.provenance.get("convt"),
                Some(BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded
                ))
            ));
            let event = result
                .fallback_events
                .iter()
                .find(|event| {
                    event.reason == CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded
                })
                .expect("explicit partial-row fallback event");
            assert!(
                event.details.contains("32/33"),
                "controller must commit only the first chunk: {}",
                event.details
            );

            let retained = result.bounds.get("convt").expect("retained target");
            let baseline = ibp.get("convt").expect("IBP target");
            for ((&lower, &upper), (&ibp_lower, &ibp_upper)) in retained
                .lower()
                .iter()
                .zip(retained.upper())
                .zip(baseline.lower().iter().zip(baseline.upper()))
            {
                assert!(lower >= ibp_lower);
                assert!(upper <= ibp_upper);
            }
            // Row 32 belonged to the rejected second wave and must remain
            // bit-identical to certified IBP.
            assert_eq!(retained.lower()[[0, 0, 32]], baseline.lower()[[0, 0, 32]]);
            assert_eq!(retained.upper()[[0, 0, 32]], baseline.upper()[[0, 0, 32]]);

            let _disabled_injected = TestChunkDeadlineGuard::after_committed_chunks(1);
            let disabled = graph
                .collect_crown_ibp_bounds_core_inner_with_cut_segment_and_policies(
                    &input,
                    ibp.clone(),
                    None,
                    None,
                    None,
                    0,
                    false,
                    PartialCrownDeadlineSalvagePolicy::Disabled,
                    super::crown_tighten::CrownIbpCollectionMode::Standard,
                )
                .expect("default-dark collector must fall back soundly");
            assert!(matches!(
                disabled.provenance.get("convt"),
                Some(BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::PerNodeDeadlineExceeded
                ))
            ));
            let disabled_bound = disabled.bounds.get("convt").expect("disabled target");
            assert_eq!(disabled_bound.lower(), baseline.lower());
            assert_eq!(disabled_bound.upper(), baseline.upper());
            assert!(
                disabled.fallback_events.iter().all(|event| {
                    event.reason != CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded
                }),
                "default-dark policy must not publish partial-row provenance"
            );
        });
    }

    /// A live finite collector authority cannot enter the legacy chunk driver,
    /// even when partial-row salvage is armed. The collector must retain the
    /// certified target box and publish no partial-row provenance.
    #[test]
    fn finite_collector_chunk_refusal_retains_ibp_without_partial_provenance() {
        ny_test_utils::env::with_env_edits(|env| {
            for key in [
                "NY_CROWN_OBJ_CHUNK",
                "NY_NO_CHUNK_WAVE_PAR",
                "NY_NO_CHUNK_ABORT",
                "NY_NO_CHUNK_GROW",
                "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
                "NY_PATCHES_BUDGET_SECS",
            ] {
                env.remove(key);
            }

            let (graph, input) = demanded_conv_transpose_33();
            let ibp = graph.collect_node_bounds(&input).expect("certified IBP");
            let baseline = ibp.get("convt").expect("IBP target");

            let retained = graph
                .collect_crown_ibp_bounds_core_inner_with_cut_segment_and_policies(
                    &input,
                    ibp.clone(),
                    Some(Instant::now() + Duration::from_secs(30)),
                    None,
                    None,
                    0,
                    false,
                    PartialCrownDeadlineSalvagePolicy::EnabledByExactEnvironment,
                    super::crown_tighten::CrownIbpCollectionMode::Standard,
                )
                .expect("finite chunk refusal must retain the sound baseline");
            assert!(matches!(
                retained.provenance.get("convt"),
                Some(BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::CrownPropagationError
                ))
            ));
            let event = retained
                .fallback_events
                .iter()
                .find(|event| event.reason == CrownIbpFallbackReason::CrownPropagationError)
                .expect("explicit finite-chunk refusal event");
            assert!(
                event.details.contains(
                    "finite objective-chunk target backward is not cooperatively bounded"
                ),
                "fallback must preserve the typed entry refusal: {}",
                event.details
            );
            let retained_bound = retained.bounds.get("convt").expect("retained target");
            assert_eq!(retained_bound.lower(), baseline.lower());
            assert_eq!(retained_bound.upper(), baseline.upper());
            assert!(
                retained.fallback_events.iter().all(|event| {
                    event.reason != CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded
                }),
                "entry refusal must not publish partially completed rows"
            );
        });
    }
}

#[cfg(test)]
mod partial_crown_chunk_salvage_tests {
    use super::{
        assemble_completed_crown_chunks, GraphTargetShapeContract,
        PartialCrownDeadlineSalvagePolicy, TargetCrownCollectionResult,
    };
    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    fn certified_ibp() -> BoundedTensor {
        BoundedTensor::new(
            arr1(&[-4.0_f32, -3.0, -2.0, -1.0]).into_dyn(),
            arr1(&[4.0_f32, 3.0, 2.0, 1.0]).into_dyn(),
        )
        .expect("certified IBP fixture")
    }

    #[test]
    fn production_gate_parser_is_exact_and_default_dark() {
        for raw in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("01"),
            Some("1 "),
            Some("NY_CROWN_DEADLINE_CHUNK_SALVAGE=1"),
        ] {
            assert_eq!(
                PartialCrownDeadlineSalvagePolicy::from_raw(raw),
                PartialCrownDeadlineSalvagePolicy::Disabled,
            );
        }
        assert_eq!(
            PartialCrownDeadlineSalvagePolicy::from_raw(Some("1")),
            PartialCrownDeadlineSalvagePolicy::EnabledByExactEnvironment,
        );
    }

    fn truncated(
        ibp: &BoundedTensor,
        chunks: Vec<(usize, Vec<f32>, Vec<f32>)>,
    ) -> (BoundedTensor, usize, usize) {
        let contract = GraphTargetShapeContract::from_bounds("target", ibp);
        match assemble_completed_crown_chunks(
            ibp,
            &contract,
            ibp.len(),
            chunks,
            Some("deterministic deadline".to_string()),
        )
        .expect("sound truncated assembly")
        {
            TargetCrownCollectionResult::DeadlineTruncated {
                bounds,
                completed_rows,
                total_rows,
                ..
            } => (bounds, completed_rows, total_rows),
            TargetCrownCollectionResult::Complete(_) => {
                panic!("deadline-truncated assembly must not claim completeness")
            }
        }
    }

    #[test]
    fn zero_completed_chunks_returns_exact_certified_ibp() {
        let ibp = certified_ibp();
        let (bounds, completed, total) = truncated(&ibp, Vec::new());
        assert_eq!((completed, total), (0, 4));
        assert_eq!(bounds.lower(), ibp.lower());
        assert_eq!(bounds.upper(), ibp.upper());
    }

    #[test]
    fn partial_completion_keeps_committed_rows_and_rejects_late_wave() {
        let ibp = certified_ibp();
        let committed = vec![(0, vec![-1.0_f32, -1.5], vec![1.0_f32, 1.5])];
        // This wave models work that returned after its deadline. Production
        // never appends it to the committed vector, even if some workers
        // happened to finish; its rows must remain exactly at certified IBP.
        let _rejected_late_wave = [(2, vec![-0.5_f32, -0.25], vec![0.5_f32, 0.25])];
        let (candidate, completed, total) = truncated(&ibp, committed);
        assert_eq!((completed, total), (2, 4));
        assert_eq!(
            candidate.lower().as_slice().expect("contiguous"),
            &[-1.0, -1.5, -2.0, -1.0]
        );
        assert_eq!(
            candidate.upper().as_slice().expect("contiguous"),
            &[1.0, 1.5, 2.0, 1.0]
        );

        // Exercise the collector's ordinary intersection seam. The retained
        // hybrid is no looser than IBP on every row and encloses the known
        // feasible fixture value zero.
        let (retained, disjoint) = ibp
            .intersection_per_element(&candidate)
            .expect("finite same-shape intersection");
        assert_eq!(disjoint, 0);
        for ((&lower, &upper), (&ibp_lower, &ibp_upper)) in retained
            .lower()
            .iter()
            .zip(retained.upper())
            .zip(ibp.lower().iter().zip(ibp.upper()))
        {
            assert!(lower >= ibp_lower);
            assert!(upper <= ibp_upper);
            assert!(lower <= 0.0 && 0.0 <= upper);
        }
        assert_eq!(
            &retained.lower().as_slice().expect("contiguous")[2..],
            &[-2.0, -1.0]
        );
        assert_eq!(
            &retained.upper().as_slice().expect("contiguous")[2..],
            &[2.0, 1.0]
        );
    }

    #[test]
    fn full_completion_is_the_only_complete_outcome() {
        let ibp = certified_ibp();
        let contract = GraphTargetShapeContract::from_bounds("target", &ibp);
        let outcome = assemble_completed_crown_chunks(
            &ibp,
            &contract,
            4,
            vec![
                (0, vec![-1.0_f32, -1.5], vec![1.0_f32, 1.5]),
                (2, vec![-0.5_f32, -0.25], vec![0.5_f32, 0.25]),
            ],
            None,
        )
        .expect("full assembly");
        let TargetCrownCollectionResult::Complete(bounds) = outcome else {
            panic!("all completed chunks must produce a complete outcome");
        };
        assert_eq!(
            bounds.lower().as_slice().expect("contiguous"),
            &[-1.0, -1.5, -0.5, -0.25]
        );
        assert_eq!(
            bounds.upper().as_slice().expect("contiguous"),
            &[1.0, 1.5, 0.5, 0.25]
        );
    }
}

/// `NY_CROWN_GAIN=1` gate, shared with the `#crown-gain` width probe.
pub(super) fn crown_gain_probe_enabled() -> bool {
    std::env::var("NY_CROWN_GAIN").ok().as_deref() == Some("1")
}

/// Which component of an accumulated CROWN relation has gone non-finite, if any.
///
/// Checked in the order the backward step touches them so the reported component
/// names the first thing to degrade: biases carry accumulated ReLU relaxation
/// slack and degrade first in practice; coefficients degrade when a matmul
/// overflows. Diagnostic only — never changes what is computed.
pub(super) fn crown_bounds_nonfinite_kind(cb: &CrownBounds) -> Option<&'static str> {
    let CrownBounds::Dense(lb) = cb else {
        // Patches relations carry their bias in the same dense lane once
        // materialized; an un-materialized patches relation has nothing to scan.
        return None;
    };
    // NaN FIRST, and reported distinctly. `LinearBounds::new_or_conservative`
    // repairs a NaN bias to -inf, so a probe that only tests `!is_finite()`
    // fires at or after that repair and structurally cannot see the origin.
    // NaN is the defect; the infinity is its (sound) shadow.
    if lb.lower_b.iter().any(|v| v.is_nan()) {
        return Some("lower_bias_NAN");
    }
    if lb.upper_b.iter().any(|v| v.is_nan()) {
        return Some("upper_bias_NAN");
    }
    if lb.lower_a.iter().any(|v| v.is_nan()) {
        return Some("lower_coeff_NAN");
    }
    if lb.upper_a.iter().any(|v| v.is_nan()) {
        return Some("upper_coeff_NAN");
    }
    if lb.lower_b.iter().any(|v| !v.is_finite()) {
        return Some("lower_bias_inf");
    }
    if lb.upper_b.iter().any(|v| !v.is_finite()) {
        return Some("upper_bias_inf");
    }
    if lb.lower_a.iter().any(|v| !v.is_finite()) {
        return Some("lower_coeff_inf");
    }
    if lb.upper_a.iter().any(|v| !v.is_finite()) {
        return Some("upper_coeff_inf");
    }
    None
}
