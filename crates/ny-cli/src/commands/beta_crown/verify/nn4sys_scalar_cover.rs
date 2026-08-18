// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact complete-cover prepass for one-free-axis, per-clause box properties.
//!
//! The NN4SYS `lindex` properties are large disjunctions of unsafe clauses.
//! Many MSCN clauses own a complete 308-coordinate box with exactly one
//! genuinely ranged coordinate; scalar LINDEX clauses are the one-coordinate
//! special case.  The existing adaptive box-refinement screen is already
//! sound, but its queue interleaves roots and descendants.  This module
//! adaptively refines a deterministic dyadic cover along the one ranged axis,
//! retains outward enclosures for every authored point coordinate, and retires
//! only leaves certified by sound batched f64 graph IBP.
//!
//! Soundness invariants:
//! - every clause must author every logical row-major input coordinate, and
//!   the complete effective box is intersected with the finite ordered global
//!   input enclosure before the clause can be admitted;
//! - authored f64 endpoints are rounded outward to f32 before promotion back
//!   to f64, matching the FLOAT network-input surface without losing points.
//!   In particular, a non-representable authored point remains a two-endpoint
//!   enclosure and is never silently replaced by a midpoint;
//! - children are `[lo, midpoint]` and `[midpoint, hi]`, share the midpoint,
//!   and are visited lower child first; an unsplittable f64 interval remains a
//!   terminal leaf rather than creating a gap;
//! - a conjunction (one unsafe clause) is impossible only when one of its own
//!   authored constraints is false under that leaf's finite ordered f64 output
//!   enclosure;
//! - no partial frontier is authoritative.  One undecided eligible leaf,
//!   propagation error, unsupported schema, cap, or deadline discards the
//!   entire prepass.  If the complete eligible subset closes, only its 1D
//!   clauses publish `true`; authenticated 0D or 2D+ clauses remain `false`;
//! - there is no SAT verdict path.  A possibly violating leaf is merely
//!   undecided and falls through to the existing attack/BaB pipeline.
//!
//! Wiring is intentionally opt-in (`NY_NN4SYS_1D_COMPLETE_COVER=1`).  The
//! publication seam is clean, but this pass duplicates the mature NN4SYS
//! screen and has not yet earned a competition-budget slice on measured
//! receipts. The rigorous terminal centered/monotonicity fallback has a
//! second exact opt-in (`NY_NN4SYS_1D_CENTERED_TERMINAL=1`) while it is being
//! measured. An optional authenticated start depth lets that stronger walk
//! prune cells before the hard cover depth; unresolved cells still split into
//! an exact cover. Every non-success outcome preserves the old path.

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ndarray::ArrayD;
use ny_core::{f64_to_f32_down, f64_to_f32_up};
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{GraphNetwork, Interval64};
use ny_tensor::BoundedTensor;
use tracing::{debug, info};

use super::BetaCrownModel;

const DEFAULT_DEPTH: u8 = 8;
const HARD_MAX_DEPTH: u8 = 24;
const DEFAULT_MAX_LEAVES: usize = 2_000_000;
const HARD_MAX_LEAVES: usize = 16_000_000;
const DEFAULT_MAX_CLAUSE_LEAF_CHECKS: usize = 4_000_000;
const HARD_MAX_CLAUSE_LEAF_CHECKS: usize = 32_000_000;
const DEFAULT_MAX_BATCH_LEAVES: usize = 512;
const HARD_MAX_BATCH_LEAVES: usize = 4096;
/// Keep the terminal MVF batch deliberately below the graph primitive's
/// internal 96-cell ceiling. The authentic 2048-wide MSCN model benefits
/// materially from batching, but every additional cell widens the
/// simultaneously live derivative channels. The opt-in may tune this up to
/// 64 while exact surface accounting and the caller's RSS limit remain active.
const DEFAULT_CENTERED_TERMINAL_BATCH_LEAVES: usize = 16;
const HARD_MAX_CENTERED_TERMINAL_BATCH_LEAVES: usize = 64;
const DEFAULT_CENTERED_TERMINAL: bool = false;
const DEFAULT_PARTIAL_COVER_BUDGET_MS: usize = 30_000;
const HARD_MAX_PARTIAL_COVER_BUDGET_MS: usize = 120_000;
/// Boundary-enclosure storage, measured as simultaneously resident f64
/// scalars.  This is not a total byte/RSS limit: collection metadata, graph
/// activations, and the graph-owned prepared Linear-weight cache are excluded.
/// The first one-leaf probe must learn the actual output size before it can
/// enforce the output-surface portion of this cap; later batches account for
/// every retained complete-box word, frontier segment endpoint, full-shape
/// Interval64 input endpoint, and output endpoint. An explicitly enabled
/// terminal centered fallback additionally reserves the simultaneously
/// returned value/centered/mono output surfaces. The graph walk evicts
/// internal tensors at last use; derivative channels and other activations
/// remain excluded like ordinary graph activations. These limitations are
/// part of why the lane remains default-off.
const DEFAULT_MAX_STORED_F64: usize = 1_048_576;
const HARD_MAX_STORED_F64: usize = 67_108_864;
const TELEMETRY_MARKER: &str = "NY_NN4SYS_1D_COMPLETE_COVER_V1";

#[derive(Debug, Clone, Copy)]
struct ScalarCoverLimits {
    depth: u8,
    max_leaves: usize,
    max_clause_leaf_checks: usize,
    max_batch_leaves: usize,
    max_stored_f64: usize,
    centered_terminal: bool,
    centered_start_depth: u8,
    centered_batch_leaves: usize,
}

impl Default for ScalarCoverLimits {
    fn default() -> Self {
        Self {
            depth: DEFAULT_DEPTH,
            max_leaves: DEFAULT_MAX_LEAVES,
            max_clause_leaf_checks: DEFAULT_MAX_CLAUSE_LEAF_CHECKS,
            max_batch_leaves: DEFAULT_MAX_BATCH_LEAVES,
            max_stored_f64: DEFAULT_MAX_STORED_F64,
            centered_terminal: DEFAULT_CENTERED_TERMINAL,
            centered_start_depth: DEFAULT_DEPTH,
            centered_batch_leaves: DEFAULT_CENTERED_TERMINAL_BATCH_LEAVES,
        }
    }
}

impl ScalarCoverLimits {
    fn from_env() -> Self {
        let defaults = Self::default();
        let depth = bounded_env_u8("NY_NN4SYS_1D_COVER_DEPTH", defaults.depth, HARD_MAX_DEPTH);
        Self {
            depth,
            max_leaves: bounded_env_usize(
                "NY_NN4SYS_1D_COVER_MAX_LEAVES",
                defaults.max_leaves,
                HARD_MAX_LEAVES,
            ),
            max_clause_leaf_checks: bounded_env_usize(
                "NY_NN4SYS_1D_COVER_MAX_CHECKS",
                defaults.max_clause_leaf_checks,
                HARD_MAX_CLAUSE_LEAF_CHECKS,
            ),
            max_batch_leaves: bounded_env_usize(
                "NY_NN4SYS_1D_COVER_BATCH",
                defaults.max_batch_leaves,
                HARD_MAX_BATCH_LEAVES,
            ),
            max_stored_f64: bounded_env_usize(
                "NY_NN4SYS_1D_COVER_MAX_STORED_F64",
                defaults.max_stored_f64,
                HARD_MAX_STORED_F64,
            ),
            centered_terminal: std::env::var("NY_NN4SYS_1D_CENTERED_TERMINAL")
                .ok()
                .as_deref()
                == Some("1"),
            centered_start_depth: bounded_env_u8(
                "NY_NN4SYS_1D_CENTERED_START_DEPTH",
                depth,
                HARD_MAX_DEPTH,
            ),
            centered_batch_leaves: bounded_env_usize(
                "NY_NN4SYS_1D_CENTERED_BATCH",
                defaults.centered_batch_leaves,
                HARD_MAX_CENTERED_TERMINAL_BATCH_LEAVES,
            ),
        }
    }

    fn valid(self) -> bool {
        self.depth <= HARD_MAX_DEPTH
            && self.centered_start_depth <= self.depth
            && self.max_leaves > 0
            && self.max_clause_leaf_checks > 0
            && self.max_batch_leaves > 0
            && self.max_stored_f64 > 0
            && (1..=HARD_MAX_CENTERED_TERMINAL_BATCH_LEAVES).contains(&self.centered_batch_leaves)
    }
}

fn bounded_env_usize(name: &str, default: usize, hard_max: usize) -> usize {
    std::env::var(name)
        .ok()
        .filter(|raw| !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&value| (1..=hard_max).contains(&value))
        .unwrap_or(default)
}

fn bounded_env_u8(name: &str, default: u8, hard_max: u8) -> u8 {
    std::env::var(name)
        .ok()
        .filter(|raw| !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|raw| raw.parse::<u8>().ok())
        .filter(|&value| value <= hard_max)
        .unwrap_or(default)
}

fn enabled() -> bool {
    std::env::var("NY_NN4SYS_1D_COMPLETE_COVER").ok().as_deref() == Some("1")
}

fn partial_cover_enabled() -> bool {
    std::env::var("NY_NN4SYS_1D_PARTIAL_COVER").ok().as_deref() == Some("1")
}

fn partial_cover_deadline(caller_deadline: Option<Instant>) -> Option<Instant> {
    let budget_ms = bounded_env_usize(
        "NY_NN4SYS_1D_PARTIAL_BUDGET_MS",
        DEFAULT_PARTIAL_COVER_BUDGET_MS,
        HARD_MAX_PARTIAL_COVER_BUDGET_MS,
    );
    let local = Instant::now().checked_add(Duration::from_millis(budget_ms as u64));
    match (caller_deadline, local) {
        (Some(caller), Some(local)) => Some(caller.min(local)),
        (caller, local) => caller.or(local),
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ScalarCoverStats {
    groups: usize,
    completed_groups: usize,
    clauses: usize,
    eligible_clauses: usize,
    excluded_non_1d_clauses: usize,
    leaves: usize,
    clause_leaf_checks: usize,
    batches: usize,
    max_depth_reached: u8,
    centered_terminal_attempts: usize,
    centered_terminal_batch_calls: usize,
    centered_terminal_refuted_clauses: usize,
    centered_terminal_failures: usize,
}

/// Deterministic witness for one obligation that remained open on the first
/// terminal dyadic cell. This is diagnostic-only: it never grants proof
/// authority, but it makes depth/correlation failures reproducible without
/// weakening the all-or-nothing publication rule.
#[derive(Debug, Clone, PartialEq, Eq)]
struct UnresolvedClauseDiagnostic {
    group_index: usize,
    clause_index: usize,
    free_axis: usize,
    segment_lower_bits: u64,
    segment_upper_bits: u64,
    depth: u8,
    /// Constraint with the largest signed falsity margin in this clause.
    /// Positive proves the constraint false; zero also proves a strict
    /// constraint false. A negative value is the remaining bound gap.
    best_constraint_index: usize,
    refutation_margin_bits: u64,
    equality_refutes: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ScalarCoverOutcome {
    /// Every admitted 1D clause was refuted over an exact cover.  The vector
    /// has property cardinality; only admitted clauses are `true`.
    Verified(ScalarCoverStats, Vec<bool>),
    /// Static admission, propagation, or resource-cap refusal.
    Declined(&'static str, ScalarCoverStats),
    /// At least one complete-cover leaf could not refute its clause.
    Incomplete(ScalarCoverStats, Vec<UnresolvedClauseDiagnostic>),
    /// The deadline was observed before all leaves were certified.
    Deadline(ScalarCoverStats),
}

impl ScalarCoverOutcome {
    fn stats(&self) -> ScalarCoverStats {
        match self {
            Self::Verified(stats, _)
            | Self::Declined(_, stats)
            | Self::Incomplete(stats, _)
            | Self::Deadline(stats) => *stats,
        }
    }
}

/// Collision-free bit-exact key for one complete effective input box.
///
/// `BTreeMap` compares all bits rather than a digest.  Signed zero therefore
/// remains distinct (conservative but deterministic), and there is no hash
/// collision seam capable of merging incompatible fixed-coordinate boxes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CompleteBoxKey {
    free_axis: usize,
    lower_bits: Box<[u64]>,
    upper_bits: Box<[u64]>,
}

impl CompleteBoxKey {
    fn input_dim(&self) -> usize {
        self.lower_bits.len()
    }

    fn lower(&self, axis: usize) -> Option<f64> {
        self.lower_bits.get(axis).copied().map(f64::from_bits)
    }

    fn upper(&self, axis: usize) -> Option<f64> {
        self.upper_bits.get(axis).copied().map(f64::from_bits)
    }
}

#[derive(Debug)]
struct ClauseGroup {
    complete_box: CompleteBoxKey,
    clauses: Vec<usize>,
}

#[derive(Debug)]
struct ClauseAdmission {
    groups: Vec<ClauseGroup>,
    eligible: Vec<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Segment {
    lower: f64,
    upper: f64,
    depth: u8,
}

#[derive(Debug, PartialEq)]
struct FrontierLeaf {
    group_index: usize,
    segment: Segment,
    /// Clauses over this root domain that no ancestor enclosure has refuted.
    /// Once a clause is refuted on a parent, that certificate covers both
    /// children and the obligation never descends again.
    obligations: Vec<usize>,
}

/// Streaming lower-before-upper DFS over a dyadic interval cover.  Only one
/// root-to-leaf path (at most `depth + 1` segments) is resident.
#[cfg(test)]
struct DyadicLeaves {
    stack: Vec<Segment>,
    target_depth: u8,
}

#[cfg(test)]
impl DyadicLeaves {
    fn new(lower: f64, upper: f64, target_depth: u8) -> Self {
        Self {
            stack: vec![Segment {
                lower,
                upper,
                depth: 0,
            }],
            target_depth,
        }
    }
}

#[cfg(test)]
impl Iterator for DyadicLeaves {
    type Item = Segment;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let segment = self.stack.pop()?;
            if segment.depth >= self.target_depth || segment.lower == segment.upper {
                return Some(segment);
            }
            let Some((lower_child, upper_child)) = split_segment(segment) else {
                // Adjacent f64 endpoints cannot be split without losing a
                // point.  The parent itself is the exact terminal cover.
                return Some(segment);
            };
            // LIFO: push upper first so the lower child is visited first.
            self.stack.push(upper_child);
            self.stack.push(lower_child);
        }
    }
}

fn split_segment(segment: Segment) -> Option<(Segment, Segment)> {
    if segment.lower == segment.upper || segment.depth == u8::MAX {
        return None;
    }
    let midpoint = f64::midpoint(segment.lower, segment.upper);
    if !(segment.lower < midpoint && midpoint < segment.upper) {
        return None;
    }
    let next_depth = segment.depth + 1;
    Some((
        Segment {
            lower: segment.lower,
            upper: midpoint,
            depth: next_depth,
        },
        Segment {
            lower: midpoint,
            upper: segment.upper,
            depth: next_depth,
        },
    ))
}

fn constraint_supported(constraint: &OutputConstraint) -> bool {
    match constraint {
        OutputConstraint::LessEq(i, j)
        | OutputConstraint::GreaterEq(i, j)
        | OutputConstraint::LessThan(i, j)
        | OutputConstraint::GreaterThan(i, j) => i != j,
        OutputConstraint::LessEqConst(_, value)
        | OutputConstraint::GreaterEqConst(_, value)
        | OutputConstraint::LessThanConst(_, value)
        | OutputConstraint::GreaterThanConst(_, value) => value.is_finite(),
        _ => false,
    }
}

fn constraint_indices_valid(constraint: &OutputConstraint, output_dim: usize) -> bool {
    match constraint {
        OutputConstraint::LessEq(i, j)
        | OutputConstraint::GreaterEq(i, j)
        | OutputConstraint::LessThan(i, j)
        | OutputConstraint::GreaterThan(i, j) => *i < output_dim && *j < output_dim,
        OutputConstraint::LessEqConst(i, _)
        | OutputConstraint::GreaterEqConst(i, _)
        | OutputConstraint::LessThanConst(i, _)
        | OutputConstraint::GreaterThanConst(i, _) => *i < output_dim,
        _ => false,
    }
}

fn standard_output_slices(output: &Interval64) -> Option<(&[f64], &[f64])> {
    if output.lower.is_empty() || output.lower.shape() != output.upper.shape() {
        return None;
    }
    // VNN-LIB output indices use logical row-major flattening. `as_slice()`
    // guarantees that order; `as_slice_memory_order()` would silently reorder
    // a non-standard-layout tensor. Unsupported layouts therefore fail open.
    let lower = output.lower.as_slice()?;
    let upper = output.upper.as_slice()?;
    lower
        .iter()
        .zip(upper)
        .all(|(&lower, &upper)| lower.is_finite() && upper.is_finite() && lower <= upper)
        .then_some((lower, upper))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ConstraintMargin {
    value: f64,
    /// Strict unsafe comparisons are already false at equality.
    equality_refutes: bool,
}

impl ConstraintMargin {
    fn refuted(self) -> bool {
        self.value > 0.0 || (self.equality_refutes && self.value >= 0.0)
    }
}

/// Signed distance to refuting one unsafe atom under an output enclosure.
/// Larger is better: positive refutes every comparison, while zero refutes
/// only strict comparisons. Admission and the first output probe authenticate
/// the schema and indices before these direct indexed accesses are used.
fn constraint_refutation_margin(
    lower: &[f64],
    upper: &[f64],
    constraint: &OutputConstraint,
) -> Option<ConstraintMargin> {
    let (value, equality_refutes) = match constraint {
        OutputConstraint::LessEqConst(index, threshold) => (lower[*index] - threshold, false),
        OutputConstraint::LessThanConst(index, threshold) => (lower[*index] - threshold, true),
        OutputConstraint::GreaterEqConst(index, threshold) => (threshold - upper[*index], false),
        OutputConstraint::GreaterThanConst(index, threshold) => (threshold - upper[*index], true),
        OutputConstraint::LessEq(i, j) => (lower[*i] - upper[*j], false),
        OutputConstraint::LessThan(i, j) => (lower[*i] - upper[*j], true),
        OutputConstraint::GreaterEq(i, j) => (lower[*j] - upper[*i], false),
        OutputConstraint::GreaterThan(i, j) => (lower[*j] - upper[*i], true),
        _ => return None,
    };
    Some(ConstraintMargin {
        value,
        equality_refutes,
    })
}

#[cfg(test)]
fn constraint_provably_false(lower: &[f64], upper: &[f64], constraint: &OutputConstraint) -> bool {
    constraint_refutation_margin(lower, upper, constraint).is_some_and(ConstraintMargin::refuted)
}

fn deadline_expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|deadline| Instant::now() >= deadline)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdmissionError {
    Declined(&'static str),
    Deadline,
}

fn build_clause_groups(
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[BTreeMap<usize, (f64, f64)>],
    max_stored_f64: usize,
    deadline: Option<Instant>,
) -> Result<ClauseAdmission, AdmissionError> {
    if clauses.is_empty() || clauses.len() != per_clause_input_bounds.len() {
        return Err(AdmissionError::Declined("clause/box cardinality mismatch"));
    }
    if input.lower().shape() != input.upper().shape() || input.lower().is_empty() {
        return Err(AdmissionError::Declined(
            "input shape is empty or lower/upper shapes differ",
        ));
    }
    // VNN-LIB X_i indexing is logical row-major flattening.  Requiring a
    // standard-layout slice prevents a memory-order reinterpretation.
    let global_lower = input.lower().as_slice().ok_or(AdmissionError::Declined(
        "global input is non-standard-layout",
    ))?;
    let global_upper = input.upper().as_slice().ok_or(AdmissionError::Declined(
        "global input is non-standard-layout",
    ))?;
    if global_lower.len() != global_upper.len()
        || global_lower
            .iter()
            .zip(global_upper)
            .any(|(&lower, &upper)| !(lower.is_finite() && upper.is_finite() && lower <= upper))
    {
        return Err(AdmissionError::Declined(
            "global input enclosure is not finite and ordered",
        ));
    }
    let input_dim = global_lower.len();
    if input_dim
        .checked_mul(2)
        .is_none_or(|words| words > max_stored_f64)
    {
        return Err(AdmissionError::Declined(
            "one complete-box boundary exceeds the memory cap",
        ));
    }

    // The complete bit-vectors themselves are the ordered map keys: grouping
    // is collision-free, deterministic, and does not retain a duplicate copy
    // of each full box.  The keys become `ClauseGroup::complete_box` below.
    let mut grouped: BTreeMap<CompleteBoxKey, Vec<usize>> = BTreeMap::new();
    let mut eligible = vec![false; clauses.len()];
    for (clause_index, (clause, clause_box)) in
        clauses.iter().zip(per_clause_input_bounds).enumerate()
    {
        if deadline_expired(deadline) {
            return Err(AdmissionError::Deadline);
        }
        if clause.is_empty() {
            return Err(AdmissionError::Declined(
                "unsupported or empty output clause",
            ));
        }
        for constraint in clause {
            if deadline_expired(deadline) {
                return Err(AdmissionError::Deadline);
            }
            if !constraint_supported(constraint) {
                return Err(AdmissionError::Declined(
                    "unsupported or empty output clause",
                ));
            }
        }
        if clause_box.len() != input_dim
            || clause_box
                .keys()
                .copied()
                .zip(0..input_dim)
                .any(|(authored, expected)| authored != expected)
        {
            return Err(AdmissionError::Declined(
                "clause box does not completely author every input coordinate",
            ));
        }

        // Admission temporarily owns one candidate full-box key alongside all
        // retained distinct keys. Refuse before allocating it if that exact
        // boundary-word surface would exceed the configured cap. This can
        // conservatively decline a duplicate at the cap, never over-claim.
        if grouped
            .len()
            .checked_add(1)
            .and_then(|boxes| boxes.checked_mul(input_dim))
            .and_then(|words| words.checked_mul(2))
            .is_none_or(|words| words > max_stored_f64)
        {
            return Err(AdmissionError::Declined(
                "complete-box admission scratch memory cap exceeded",
            ));
        }
        let mut effective_lower = Vec::with_capacity(input_dim);
        let mut effective_upper = Vec::with_capacity(input_dim);
        let mut genuinely_varying_axis = None;
        let mut genuinely_varying_count = 0usize;
        for (axis, (&global_lower, &global_upper)) in
            global_lower.iter().zip(global_upper).enumerate()
        {
            if deadline_expired(deadline) {
                return Err(AdmissionError::Deadline);
            }
            let &(authored_lower, authored_upper) =
                clause_box.get(&axis).ok_or(AdmissionError::Declined(
                    "clause box does not completely author every input coordinate",
                ))?;
            if !(authored_lower.is_finite()
                && authored_upper.is_finite()
                && authored_lower <= authored_upper)
            {
                return Err(AdmissionError::Declined(
                    "clause input interval is not finite and ordered",
                ));
            }
            if authored_lower < authored_upper {
                genuinely_varying_count = genuinely_varying_count.saturating_add(1);
                genuinely_varying_axis = Some(axis);
            }

            // The network's input element type is FLOAT. Round the exact
            // VNN-LIB endpoint outward once, intersect with the already-sound
            // global FLOAT enclosure, then stay in f64 for proof. Authored
            // points deliberately keep both rounded endpoints.
            let lower = f64::from(f64_to_f32_down(authored_lower)).max(f64::from(global_lower));
            let upper = f64::from(f64_to_f32_up(authored_upper)).min(f64::from(global_upper));
            if !(lower.is_finite() && upper.is_finite() && lower <= upper) {
                return Err(AdmissionError::Declined(
                    "directed clause/global intersection is empty or non-finite",
                ));
            }
            effective_lower.push(lower.to_bits());
            effective_upper.push(upper.to_bits());
        }

        // Zero-dimensional and 2D+ boxes are fully authenticated but outside
        // this lane.  They stay false in a successful subset publication.
        if genuinely_varying_count != 1 {
            continue;
        }
        let free_axis = genuinely_varying_axis.expect("one varying axis was counted");
        let free_lower = f64::from_bits(effective_lower[free_axis]);
        let free_upper = f64::from_bits(effective_upper[free_axis]);
        if free_lower >= free_upper {
            return Err(AdmissionError::Declined(
                "genuinely ranged coordinate collapsed after effective intersection",
            ));
        }
        let key = CompleteBoxKey {
            free_axis,
            lower_bits: effective_lower.into_boxed_slice(),
            upper_bits: effective_upper.into_boxed_slice(),
        };
        eligible[clause_index] = true;
        grouped.entry(key).or_default().push(clause_index);

        // Each distinct key retains exactly two u64 boundary words per input
        // coordinate. Count them as f64-sized storage before admitting more.
        if grouped
            .len()
            .checked_mul(input_dim)
            .and_then(|words| words.checked_mul(2))
            .is_none_or(|words| words > max_stored_f64)
        {
            return Err(AdmissionError::Declined(
                "complete-box boundary memory cap exceeded during admission",
            ));
        }
    }
    if deadline_expired(deadline) {
        return Err(AdmissionError::Deadline);
    }
    if grouped.is_empty() {
        return Err(AdmissionError::Declined(
            "no exactly-one-varying-axis clauses",
        ));
    }
    let groups = grouped
        .into_iter()
        .map(|(complete_box, clauses)| ClauseGroup {
            complete_box,
            clauses,
        })
        .collect();
    Ok(ClauseAdmission { groups, eligible })
}

#[derive(Debug, PartialEq, Eq)]
enum BatchOutcome {
    Complete {
        output_dim: usize,
        /// One result per live obligation on each input leaf. A refuted
        /// obligation is certified over both future children; an open result
        /// retains the best signed margin for deterministic terminal telemetry.
        clause_evaluations: Vec<Vec<ClauseEvaluation>>,
    },
    Deadline,
    Declined(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClauseEvaluation {
    refuted: bool,
    best_constraint_index: usize,
    refutation_margin_bits: u64,
    equality_refutes: bool,
}

enum LeafEvaluationOutcome {
    Complete(Vec<ClauseEvaluation>),
    Deadline,
    Declined(&'static str),
}

fn evaluate_leaf_output(
    output: &Interval64,
    leaf: &FrontierLeaf,
    clauses: &[Vec<OutputConstraint>],
    expected_output_dim: usize,
    deadline: Option<Instant>,
) -> LeafEvaluationOutcome {
    if output.lower.len() != expected_output_dim || output.upper.len() != expected_output_dim {
        return LeafEvaluationOutcome::Declined("f64 output shape changed across leaves");
    }
    let Some((lower, upper)) = standard_output_slices(output) else {
        return LeafEvaluationOutcome::Declined(
            "f64 output is non-finite, inverted, or non-standard-layout",
        );
    };
    let mut evaluations = Vec::with_capacity(leaf.obligations.len());
    for &clause_index in &leaf.obligations {
        if deadline_expired(deadline) {
            return LeafEvaluationOutcome::Deadline;
        }
        let Some(clause) = clauses.get(clause_index) else {
            return LeafEvaluationOutcome::Declined("internal clause index mismatch");
        };
        let mut best = None;
        let mut refuted = false;
        for (constraint_index, constraint) in clause.iter().enumerate() {
            if deadline_expired(deadline) {
                return LeafEvaluationOutcome::Deadline;
            }
            if !constraint_indices_valid(constraint, expected_output_dim) {
                return LeafEvaluationOutcome::Declined(
                    "output constraint index is outside graph output",
                );
            }
            let Some(margin) = constraint_refutation_margin(lower, upper, constraint) else {
                return LeafEvaluationOutcome::Declined(
                    "unsupported output constraint reached batch evaluation",
                );
            };
            if best
                .as_ref()
                .is_none_or(|(_, current): &(usize, ConstraintMargin)| {
                    margin.value.total_cmp(&current.value).is_gt()
                })
            {
                best = Some((constraint_index, margin));
            }
            if margin.refuted() {
                refuted = true;
                break;
            }
        }
        let Some((best_constraint_index, best_margin)) = best else {
            return LeafEvaluationOutcome::Declined("empty clause reached batch evaluation");
        };
        evaluations.push(ClauseEvaluation {
            refuted,
            best_constraint_index,
            refutation_margin_bits: best_margin.value.to_bits(),
            equality_refutes: best_margin.equality_refutes,
        });
    }
    LeafEvaluationOutcome::Complete(evaluations)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_batch(
    graph: &GraphNetwork,
    inputs: &[Interval64],
    leaves: &[FrontierLeaf],
    clauses: &[Vec<OutputConstraint>],
    expected_output_dim: Option<usize>,
    limits: ScalarCoverLimits,
    weights: Option<&ny_propagate::F64WeightCache>,
    deadline: Option<Instant>,
) -> BatchOutcome {
    if deadline_expired(deadline) {
        return BatchOutcome::Deadline;
    }
    let outputs =
        match graph.propagate_ibp_f64_cells_cached_with_deadline(inputs, weights, deadline) {
            Ok(outputs) => outputs,
            Err(error) if error.is_deadline_exceeded() => return BatchOutcome::Deadline,
            Err(_) => return BatchOutcome::Declined("batched f64 graph propagation failed"),
        };
    if deadline_expired(deadline) {
        return BatchOutcome::Deadline;
    }
    if outputs.len() != inputs.len() || outputs.len() != leaves.len() || outputs.is_empty() {
        return BatchOutcome::Declined("batched f64 output cardinality mismatch");
    }

    let output_dim = outputs[0].lower.len();
    if output_dim == 0
        || expected_output_dim.is_some_and(|expected| expected != output_dim)
        || outputs
            .iter()
            .any(|output| output.lower.len() != output_dim || output.upper.len() != output_dim)
    {
        return BatchOutcome::Declined("batched f64 output shape changed across leaves");
    }
    let per_leaf_stored = inputs[0].lower.len().checked_mul(2).and_then(|input| {
        output_dim
            .checked_mul(2)
            .and_then(|output| input.checked_add(output))
    });
    if per_leaf_stored
        .and_then(|per_leaf| per_leaf.checked_mul(inputs.len()))
        .is_none_or(|stored| stored > limits.max_stored_f64)
    {
        return BatchOutcome::Declined("f64 boundary-enclosure memory cap exceeded");
    }
    if expected_output_dim.is_none() {
        for clause in clauses {
            for constraint in clause {
                if deadline_expired(deadline) {
                    return BatchOutcome::Deadline;
                }
                if !constraint_indices_valid(constraint, output_dim) {
                    return BatchOutcome::Declined(
                        "output constraint index is outside graph output",
                    );
                }
            }
        }
    }

    let mut clause_evaluations = Vec::with_capacity(outputs.len());
    for (leaf, output) in leaves.iter().zip(&outputs) {
        match evaluate_leaf_output(output, leaf, clauses, output_dim, deadline) {
            LeafEvaluationOutcome::Complete(evaluations) => clause_evaluations.push(evaluations),
            LeafEvaluationOutcome::Deadline => return BatchOutcome::Deadline,
            LeafEvaluationOutcome::Declined(reason) => return BatchOutcome::Declined(reason),
        }
    }
    BatchOutcome::Complete {
        output_dim,
        clause_evaluations,
    }
}

fn one_axis_input_interval(
    input: &BoundedTensor,
    group: &ClauseGroup,
    segment: Segment,
) -> Option<Interval64> {
    let complete_box = &group.complete_box;
    if complete_box.input_dim() != input.lower().len()
        || complete_box.lower_bits.len() != complete_box.upper_bits.len()
        || complete_box.free_axis >= complete_box.input_dim()
    {
        return None;
    }
    let mut lower: Vec<f64> = complete_box
        .lower_bits
        .iter()
        .copied()
        .map(f64::from_bits)
        .collect();
    let mut upper: Vec<f64> = complete_box
        .upper_bits
        .iter()
        .copied()
        .map(f64::from_bits)
        .collect();
    lower[complete_box.free_axis] = segment.lower;
    upper[complete_box.free_axis] = segment.upper;
    Some(Interval64 {
        lower: ArrayD::from_shape_vec(input.lower().raw_dim(), lower).ok()?,
        upper: ArrayD::from_shape_vec(input.upper().raw_dim(), upper).ok()?,
    })
}

fn terminal_diagnostics(
    groups: &[ClauseGroup],
    group_index: usize,
    segment: Segment,
    unresolved: &[(usize, ClauseEvaluation)],
) -> Option<Vec<UnresolvedClauseDiagnostic>> {
    let group = groups.get(group_index)?;
    (!unresolved.is_empty()).then(|| {
        unresolved
            .iter()
            .map(|&(clause_index, evaluation)| UnresolvedClauseDiagnostic {
                group_index,
                clause_index,
                free_axis: group.complete_box.free_axis,
                segment_lower_bits: segment.lower.to_bits(),
                segment_upper_bits: segment.upper.to_bits(),
                depth: segment.depth,
                best_constraint_index: evaluation.best_constraint_index,
                refutation_margin_bits: evaluation.refutation_margin_bits,
                equality_refutes: evaluation.equality_refutes,
            })
            .collect()
    })
}

enum CenteredTerminalOutcome {
    Complete(Vec<ClauseEvaluation>),
    Deadline,
    Failed,
}

enum CenteredTerminalBatchOutcome {
    Complete(Vec<Vec<ClauseEvaluation>>),
    Deadline,
    Failed,
}

fn merge_clause_evaluations(
    basic: &[ClauseEvaluation],
    centered: Vec<ClauseEvaluation>,
) -> Option<Vec<ClauseEvaluation>> {
    (basic.len() == centered.len()).then(|| {
        basic
            .iter()
            .copied()
            .zip(centered)
            .map(|(basic, centered)| {
                if basic.refuted {
                    basic
                } else if centered.refuted
                    || f64::from_bits(centered.refutation_margin_bits)
                        .total_cmp(&f64::from_bits(basic.refutation_margin_bits))
                        .is_gt()
                {
                    centered
                } else {
                    basic
                }
            })
            .collect()
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_centered_terminal(
    graph: &GraphNetwork,
    input: &Interval64,
    leaf: &FrontierLeaf,
    clauses: &[Vec<OutputConstraint>],
    output_dim: usize,
    weights: Option<&ny_propagate::F64WeightCache>,
    deadline: Option<Instant>,
) -> CenteredTerminalOutcome {
    if deadline_expired(deadline) {
        return CenteredTerminalOutcome::Deadline;
    }
    let mut outputs = match graph.propagate_ibp_f64_centered_mono_cells_cached_with_deadline(
        std::slice::from_ref(input),
        true,
        weights,
        deadline,
    ) {
        Ok(outputs) if outputs.len() == 1 => outputs,
        Ok(_) => return CenteredTerminalOutcome::Failed,
        Err(error) if error.is_deadline_exceeded() => return CenteredTerminalOutcome::Deadline,
        Err(_) => return CenteredTerminalOutcome::Failed,
    };
    if deadline_expired(deadline) {
        return CenteredTerminalOutcome::Deadline;
    }
    let output = outputs
        .pop()
        .expect("one centered output was authenticated");
    let strongest = output.mono.as_ref().unwrap_or(&output.centered);
    match evaluate_leaf_output(strongest, leaf, clauses, output_dim, deadline) {
        LeafEvaluationOutcome::Complete(evaluations) => {
            CenteredTerminalOutcome::Complete(evaluations)
        }
        LeafEvaluationOutcome::Deadline => CenteredTerminalOutcome::Deadline,
        LeafEvaluationOutcome::Declined(_) => CenteredTerminalOutcome::Failed,
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_centered_terminal_batch(
    graph: &GraphNetwork,
    batch_inputs: &[Interval64],
    leaves: &[FrontierLeaf],
    candidate_indices: &[usize],
    clauses: &[Vec<OutputConstraint>],
    output_dim: usize,
    weights: Option<&ny_propagate::F64WeightCache>,
    deadline: Option<Instant>,
) -> CenteredTerminalBatchOutcome {
    if deadline_expired(deadline) {
        return CenteredTerminalBatchOutcome::Deadline;
    }
    if candidate_indices.len() < 2 {
        return CenteredTerminalBatchOutcome::Failed;
    }
    let mut selected_inputs = Vec::with_capacity(candidate_indices.len());
    for &index in candidate_indices {
        let (Some(input), Some(_)) = (batch_inputs.get(index), leaves.get(index)) else {
            return CenteredTerminalBatchOutcome::Failed;
        };
        selected_inputs.push(input.clone());
    }
    let outputs = match graph.propagate_ibp_f64_centered_mono_cells_cached_with_deadline(
        &selected_inputs,
        true,
        weights,
        deadline,
    ) {
        Ok(outputs) if outputs.len() == candidate_indices.len() => outputs,
        Ok(_) => return CenteredTerminalBatchOutcome::Failed,
        Err(error) if error.is_deadline_exceeded() => {
            return CenteredTerminalBatchOutcome::Deadline
        }
        Err(_) => return CenteredTerminalBatchOutcome::Failed,
    };
    if deadline_expired(deadline) {
        return CenteredTerminalBatchOutcome::Deadline;
    }

    let mut evaluations = Vec::with_capacity(outputs.len());
    for (&index, output) in candidate_indices.iter().zip(outputs) {
        let Some(leaf) = leaves.get(index) else {
            return CenteredTerminalBatchOutcome::Failed;
        };
        let strongest = output.mono.as_ref().unwrap_or(&output.centered);
        match evaluate_leaf_output(strongest, leaf, clauses, output_dim, deadline) {
            LeafEvaluationOutcome::Complete(candidate) => evaluations.push(candidate),
            LeafEvaluationOutcome::Deadline => return CenteredTerminalBatchOutcome::Deadline,
            LeafEvaluationOutcome::Declined(_) => return CenteredTerminalBatchOutcome::Failed,
        }
    }
    CenteredTerminalBatchOutcome::Complete(evaluations)
}

#[cfg(test)]
fn run_complete_scalar_cover(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[BTreeMap<usize, (f64, f64)>],
    limits: ScalarCoverLimits,
    deadline: Option<Instant>,
) -> ScalarCoverOutcome {
    let mut completed_clauses = vec![false; clauses.len()];
    run_complete_scalar_cover_recording(
        graph,
        input,
        clauses,
        per_clause_input_bounds,
        limits,
        deadline,
        &mut completed_clauses,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_complete_scalar_cover_recording(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[BTreeMap<usize, (f64, f64)>],
    limits: ScalarCoverLimits,
    deadline: Option<Instant>,
    completed_clauses: &mut [bool],
) -> ScalarCoverOutcome {
    let mut stats = ScalarCoverStats {
        clauses: clauses.len(),
        ..ScalarCoverStats::default()
    };
    if completed_clauses.len() != clauses.len() {
        return ScalarCoverOutcome::Declined("partial certificate cardinality mismatch", stats);
    }
    completed_clauses.fill(false);
    if !limits.valid() {
        return ScalarCoverOutcome::Declined("invalid cover limits", stats);
    }
    if deadline_expired(deadline) {
        return ScalarCoverOutcome::Deadline(stats);
    }
    if !graph.supports_ibp_f64_cell() {
        return ScalarCoverOutcome::Declined("graph lacks complete f64 IBP support", stats);
    }
    let admission = match build_clause_groups(
        input,
        clauses,
        per_clause_input_bounds,
        limits.max_stored_f64,
        deadline,
    ) {
        Ok(admission) => admission,
        Err(AdmissionError::Declined(reason)) => {
            return ScalarCoverOutcome::Declined(reason, stats)
        }
        Err(AdmissionError::Deadline) => return ScalarCoverOutcome::Deadline(stats),
    };
    let ClauseAdmission { groups, eligible } = admission;
    stats.eligible_clauses = eligible.iter().filter(|&&eligible| eligible).count();
    stats.excluded_non_1d_clauses = clauses.len() - stats.eligible_clauses;
    if stats.eligible_clauses > limits.max_clause_leaf_checks {
        return ScalarCoverOutcome::Declined("clause-leaf check cap exceeded", stats);
    }
    stats.groups = groups.len();

    // DFS stack in reverse bit-exact key order. Popping visits groups in stable
    // BTree order; unresolved parents push upper then lower, so the lower child
    // is always next. Certified leaves are retired immediately. Thus the stack
    // plus retired certificates is an exact cover at every transition, without
    // paying a uniform 2^depth grid for already-easy regions.
    let mut frontier: Vec<FrontierLeaf> = groups
        .iter()
        .enumerate()
        .rev()
        .map(|(group_index, group)| FrontierLeaf {
            group_index,
            segment: Segment {
                lower: group
                    .complete_box
                    .lower(group.complete_box.free_axis)
                    .expect("authenticated free lower endpoint"),
                upper: group
                    .complete_box
                    .upper(group.complete_box.free_axis)
                    .expect("authenticated free upper endpoint"),
                depth: 0,
            },
            obligations: group.clauses.clone(),
        })
        .collect();
    let mut frontier_checks = stats.eligible_clauses;
    // Diagnostic-only exact accounting: every group starts with one live root.
    // Refuting a leaf removes one; splitting replaces one with two. A zero
    // count therefore means the group's entire one-dimensional cover retired.
    let mut outstanding_group_leaves = vec![1usize; groups.len()];
    let input_dim = input.lower().len();
    let group_storage = match groups
        .len()
        .checked_mul(input_dim)
        .and_then(|words| words.checked_mul(2))
    {
        Some(stored) => stored,
        None => return ScalarCoverOutcome::Declined("group storage count overflow", stats),
    };
    if frontier.len() > limits.max_leaves
        || frontier_checks > limits.max_clause_leaf_checks
        || frontier
            .len()
            .checked_mul(2)
            .and_then(|frontier| group_storage.checked_add(frontier))
            .is_none_or(|stored| stored > limits.max_stored_f64)
    {
        return ScalarCoverOutcome::Declined("initial frontier cap exceeded", stats);
    }
    let mut output_dim: Option<usize> = None;
    // The 2048-wide MSCN Linears are much faster when a multi-leaf batch reuses
    // exact prepared W^T/|W^T| tensors. Build them only after the one-leaf
    // output-shape probe succeeds, and fence the atomic build on both sides of
    // the caller deadline. This cache changes no bound bits.
    let weight_cache = OnceLock::new();

    while !frontier.is_empty() {
        if deadline_expired(deadline) {
            return ScalarCoverOutcome::Deadline(stats);
        }
        // `2*input_dim*groups + 2*frontier` counts every retained complete-box
        // boundary word and live segment endpoint. A popped leaf moves into
        // `work`, so its two segment endpoints remain part of this resident
        // count. Its full-shape Interval64 adds `2*input_dim` f64s and its
        // output adds `2*output_dim`. The first one-leaf probe discovers the
        // output size; subsequent batches use exact boundary-surface counts.
        let resident_frontier = match frontier
            .len()
            .checked_mul(2)
            .and_then(|frontier| group_storage.checked_add(frontier))
        {
            Some(stored) if stored <= limits.max_stored_f64 => stored,
            _ => {
                return ScalarCoverOutcome::Declined(
                    "f64 boundary-enclosure memory cap exceeded",
                    stats,
                )
            }
        };
        let batch_len = if let Some(output_dim) = output_dim {
            let added_f64_per_leaf = match input_dim.checked_mul(2).and_then(|input| {
                output_dim
                    .checked_mul(2)
                    .and_then(|output| input.checked_add(output))
            }) {
                Some(value) if value > 0 => value,
                _ => {
                    return ScalarCoverOutcome::Declined(
                        "invalid f64 output storage requirement",
                        stats,
                    )
                }
            };
            let by_memory = (limits.max_stored_f64 - resident_frontier) / added_f64_per_leaf;
            frontier.len().min(limits.max_batch_leaves).min(by_memory)
        } else {
            // One output element is the minimum valid surface. The first
            // probe therefore needs `2*input_dim + 2` boundary f64s beyond
            // the retained complete boxes and frontier segments.
            let Some(minimum_probe) = input_dim
                .checked_mul(2)
                .and_then(|input| input.checked_add(2))
            else {
                return ScalarCoverOutcome::Declined(
                    "invalid f64 input storage requirement",
                    stats,
                );
            };
            if limits.max_stored_f64 - resident_frontier < minimum_probe {
                return ScalarCoverOutcome::Declined(
                    "f64 boundary-enclosure memory cap admits no leaf",
                    stats,
                );
            }
            1
        };
        if batch_len == 0 {
            return ScalarCoverOutcome::Declined(
                "f64 boundary-enclosure memory cap admits no leaf",
                stats,
            );
        }

        let mut work = Vec::with_capacity(batch_len);
        let mut batch_inputs = Vec::with_capacity(batch_len);
        for _ in 0..batch_len {
            let leaf = frontier.pop().expect("batch length is bounded by frontier");
            let obligations = leaf.obligations.len();
            frontier_checks = match frontier_checks.checked_sub(obligations) {
                Some(checks) => checks,
                None => {
                    return ScalarCoverOutcome::Declined(
                        "internal frontier obligation underflow",
                        stats,
                    )
                }
            };
            stats.leaves = match stats.leaves.checked_add(1) {
                Some(leaves) if leaves <= limits.max_leaves => leaves,
                _ => return ScalarCoverOutcome::Declined("leaf cap exceeded", stats),
            };
            stats.clause_leaf_checks = match stats.clause_leaf_checks.checked_add(obligations) {
                Some(checks) if checks <= limits.max_clause_leaf_checks => checks,
                _ => return ScalarCoverOutcome::Declined("clause-leaf check cap exceeded", stats),
            };
            stats.max_depth_reached = stats.max_depth_reached.max(leaf.segment.depth);
            let Some(group) = groups.get(leaf.group_index) else {
                return ScalarCoverOutcome::Declined("internal group index mismatch", stats);
            };
            let Some(interval) = one_axis_input_interval(input, group, leaf.segment) else {
                return ScalarCoverOutcome::Declined(
                    "failed to construct full-shape one-axis input leaf",
                    stats,
                );
            };
            batch_inputs.push(interval);
            work.push(leaf);
        }

        stats.batches += 1;
        let weights = if batch_len >= 2 && graph.f64_batch_worthwhile() {
            if deadline_expired(deadline) {
                return ScalarCoverOutcome::Deadline(stats);
            }
            let weights = weight_cache.get_or_init(|| graph.build_f64_weight_cache());
            if deadline_expired(deadline) {
                return ScalarCoverOutcome::Deadline(stats);
            }
            Some(weights)
        } else {
            None
        };
        match evaluate_batch(
            graph,
            &batch_inputs,
            &work,
            clauses,
            output_dim,
            limits,
            weights,
            deadline,
        ) {
            BatchOutcome::Complete {
                output_dim: observed,
                mut clause_evaluations,
            } => {
                if clause_evaluations.len() != work.len() {
                    return ScalarCoverOutcome::Declined(
                        "internal batch verdict cardinality mismatch",
                        stats,
                    );
                }
                let first_probe_storage = input_dim
                    .checked_mul(2)
                    .and_then(|input| {
                        observed
                            .checked_mul(2)
                            .and_then(|output| input.checked_add(output))
                    })
                    .and_then(|added| resident_frontier.checked_add(added));
                if output_dim.is_none()
                    && first_probe_storage.is_none_or(|stored| stored > limits.max_stored_f64)
                {
                    return ScalarCoverOutcome::Declined(
                        "f64 boundary-enclosure memory cap exceeded",
                        stats,
                    );
                }
                output_dim = Some(observed);
                if work
                    .iter()
                    .zip(&clause_evaluations)
                    .any(|(leaf, evaluations)| evaluations.len() != leaf.obligations.len())
                {
                    return ScalarCoverOutcome::Declined(
                        "internal clause verdict cardinality mismatch",
                        stats,
                    );
                }

                // The authentic MSCN pilot reached many eligible cells in one
                // basic-IBP frontier batch, but previously launched the much
                // stronger centered/monotonicity walk once per cell. Collect
                // the deterministic reachable prefix at or beyond the opted-in
                // start depth and pay one sound batched graph walk. Centered-
                // refuted cells are pruned; unresolved nonterminal cells keep
                // splitting, so the final certificate is still an exact cover.
                // Stop at an ineligible terminal because the DFS must return an
                // incomplete diagnostic there. A batch failure falls back to
                // the existing per-cell path; a deadline remains authoritative.
                if limits.centered_terminal {
                    let mut candidates = Vec::new();
                    for (work_index, (leaf, evaluations)) in
                        work.iter().zip(&clause_evaluations).enumerate()
                    {
                        let terminal = leaf.segment.depth >= limits.depth
                            || split_segment(leaf.segment).is_none();
                        if evaluations.iter().all(|evaluation| evaluation.refuted) {
                            continue;
                        }
                        if leaf.segment.depth < limits.centered_start_depth {
                            if terminal {
                                break;
                            }
                            continue;
                        }
                        let Some(group) = groups.get(leaf.group_index) else {
                            return ScalarCoverOutcome::Declined(
                                "internal group index mismatch",
                                stats,
                            );
                        };
                        if !graph.ibp_f64_centered_only_seeds_axis(
                            &batch_inputs[work_index],
                            group.complete_box.free_axis,
                        ) {
                            if terminal {
                                break;
                            }
                            continue;
                        }
                        candidates.push(work_index);
                        if candidates.len() >= limits.centered_batch_leaves {
                            break;
                        }
                    }

                    if candidates.len() >= 2 {
                        // `batch_inputs` remains live. Selecting a sparse
                        // terminal prefix clones those complete input surfaces;
                        // the returned value/centered/mono result retains six
                        // output endpoints per cell. Graph activations remain
                        // outside this boundary-surface cap, as documented on
                        // `DEFAULT_MAX_STORED_F64`.
                        let centered_batch_surface_fits = batch_inputs
                            .len()
                            .checked_mul(input_dim)
                            .and_then(|words| words.checked_mul(2))
                            .and_then(|retained_inputs| {
                                candidates
                                    .len()
                                    .checked_mul(input_dim)
                                    .and_then(|words| words.checked_mul(2))
                                    .and_then(|selected_inputs| {
                                        retained_inputs.checked_add(selected_inputs)
                                    })
                            })
                            .and_then(|inputs| {
                                candidates
                                    .len()
                                    .checked_mul(observed)
                                    .and_then(|words| words.checked_mul(6))
                                    .and_then(|outputs| inputs.checked_add(outputs))
                            })
                            .and_then(|added| resident_frontier.checked_add(added))
                            .is_some_and(|stored| stored <= limits.max_stored_f64);
                        if centered_batch_surface_fits {
                            stats.centered_terminal_attempts = stats
                                .centered_terminal_attempts
                                .saturating_add(candidates.len());
                            stats.centered_terminal_batch_calls =
                                stats.centered_terminal_batch_calls.saturating_add(1);
                            match evaluate_centered_terminal_batch(
                                graph,
                                &batch_inputs,
                                &work,
                                &candidates,
                                clauses,
                                observed,
                                weight_cache.get(),
                                deadline,
                            ) {
                                CenteredTerminalBatchOutcome::Complete(centered_batch) => {
                                    if centered_batch.len() != candidates.len() {
                                        return ScalarCoverOutcome::Declined(
                                            "centered terminal batch verdict cardinality mismatch",
                                            stats,
                                        );
                                    }
                                    for (&work_index, centered) in
                                        candidates.iter().zip(centered_batch)
                                    {
                                        let evaluations = &mut clause_evaluations[work_index];
                                        let newly_refuted = evaluations
                                            .iter()
                                            .zip(&centered)
                                            .filter(|(basic, centered)| {
                                                !basic.refuted && centered.refuted
                                            })
                                            .count();
                                        let Some(merged) =
                                            merge_clause_evaluations(evaluations, centered)
                                        else {
                                            return ScalarCoverOutcome::Declined(
                                                "centered terminal batch verdict cardinality mismatch",
                                                stats,
                                            );
                                        };
                                        stats.centered_terminal_refuted_clauses = stats
                                            .centered_terminal_refuted_clauses
                                            .saturating_add(newly_refuted);
                                        *evaluations = merged;
                                    }
                                }
                                CenteredTerminalBatchOutcome::Deadline => {
                                    return ScalarCoverOutcome::Deadline(stats)
                                }
                                CenteredTerminalBatchOutcome::Failed => {
                                    stats.centered_terminal_failures = stats
                                        .centered_terminal_failures
                                        .saturating_add(candidates.len());
                                }
                            }
                        }
                    }
                }

                // Inspect the popped batch in its actual DFS order first, so
                // a terminal diagnostic names the deterministic first open
                // leaf rather than the last item needed by reverse pushing.
                for (work_index, (leaf, evaluations)) in
                    work.iter().zip(&mut clause_evaluations).enumerate()
                {
                    if deadline_expired(deadline) {
                        return ScalarCoverOutcome::Deadline(stats);
                    }
                    if evaluations.len() != leaf.obligations.len() {
                        return ScalarCoverOutcome::Declined(
                            "internal clause verdict cardinality mismatch",
                            stats,
                        );
                    }
                    let terminal =
                        leaf.segment.depth >= limits.depth || split_segment(leaf.segment).is_none();
                    if evaluations.iter().all(|evaluation| evaluation.refuted) {
                        continue;
                    }

                    // Exact secondary opt-in: only the authenticated authored
                    // free axis may seed the rigorous centered derivative
                    // walk. Narrow outward enclosures of authored point
                    // coordinates remain in its zeroth-order center box; they
                    // are never collapsed or discarded. The monotonicity-
                    // corner result, when available, is already intersected
                    // with the centered enclosure. Either independent sound
                    // enclosure may refute an atom; no sampled derivative or
                    // heuristic sign is consumed.
                    let authenticated_free_axis = groups
                        .get(leaf.group_index)
                        .map(|group| group.complete_box.free_axis);
                    let centered_ready = limits.centered_terminal
                        && leaf.segment.depth >= limits.centered_start_depth
                        && authenticated_free_axis.is_some_and(|free_axis| {
                            graph.ibp_f64_centered_only_seeds_axis(
                                &batch_inputs[work_index],
                                free_axis,
                            )
                        });
                    if !terminal && !centered_ready {
                        continue;
                    }
                    // Nonterminal cells that miss the batching threshold simply
                    // split: a single-cell centered walk there is redundant and
                    // was expensive on the authentic MSCN model. At the hard
                    // terminal, retain the single-cell path as the final chance
                    // to close the exact-cover leaf (including after a batch,
                    // whose different rounding path can be incomparable).
                    if centered_ready && terminal {
                        // All basic outputs have been dropped by
                        // `evaluate_batch`, but every batch input remains.
                        // Reserve worst-case value + centered + mono output
                        // endpoints for the one terminal fallback.
                        let centered_surface_fits = batch_inputs
                            .len()
                            .checked_mul(input_dim)
                            .and_then(|words| words.checked_mul(2))
                            .and_then(|inputs| {
                                observed
                                    .checked_mul(6)
                                    .and_then(|outputs| inputs.checked_add(outputs))
                            })
                            .and_then(|added| resident_frontier.checked_add(added))
                            .is_some_and(|stored| stored <= limits.max_stored_f64);
                        if centered_surface_fits {
                            stats.centered_terminal_attempts =
                                stats.centered_terminal_attempts.saturating_add(1);
                            match evaluate_centered_terminal(
                                graph,
                                &batch_inputs[work_index],
                                leaf,
                                clauses,
                                observed,
                                weight_cache.get(),
                                deadline,
                            ) {
                                CenteredTerminalOutcome::Complete(centered) => {
                                    let newly_refuted = evaluations
                                        .iter()
                                        .zip(&centered)
                                        .filter(|(basic, centered)| {
                                            !basic.refuted && centered.refuted
                                        })
                                        .count();
                                    let Some(merged) =
                                        merge_clause_evaluations(evaluations, centered)
                                    else {
                                        return ScalarCoverOutcome::Declined(
                                            "centered terminal verdict cardinality mismatch",
                                            stats,
                                        );
                                    };
                                    stats.centered_terminal_refuted_clauses = stats
                                        .centered_terminal_refuted_clauses
                                        .saturating_add(newly_refuted);
                                    *evaluations = merged;
                                    if evaluations.iter().all(|evaluation| evaluation.refuted) {
                                        continue;
                                    }
                                }
                                CenteredTerminalOutcome::Deadline => {
                                    return ScalarCoverOutcome::Deadline(stats)
                                }
                                CenteredTerminalOutcome::Failed => {
                                    stats.centered_terminal_failures =
                                        stats.centered_terminal_failures.saturating_add(1);
                                }
                            }
                        }
                    }
                    if !terminal {
                        continue;
                    }
                    let unresolved: Vec<_> = leaf
                        .obligations
                        .iter()
                        .copied()
                        .zip(evaluations.iter().copied())
                        .filter(|(_, evaluation)| !evaluation.refuted)
                        .collect();
                    let Some(diagnostics) =
                        terminal_diagnostics(&groups, leaf.group_index, leaf.segment, &unresolved)
                    else {
                        return ScalarCoverOutcome::Declined(
                            "failed to construct terminal diagnostic",
                            stats,
                        );
                    };
                    return ScalarCoverOutcome::Incomplete(stats, diagnostics);
                }

                // Basic/centered outputs have been reduced to compact
                // per-obligation records. Release duplicated inputs before
                // growing the next exact-cover frontier.
                drop(batch_inputs);

                // Reverse the popped batch before pushing: this preserves the
                // original DFS order across multiple unresolved parents.
                for (mut leaf, evaluations) in work.into_iter().zip(clause_evaluations).rev() {
                    if deadline_expired(deadline) {
                        return ScalarCoverOutcome::Deadline(stats);
                    }
                    let unresolved: Vec<_> = leaf
                        .obligations
                        .iter()
                        .copied()
                        .zip(evaluations)
                        .filter(|(_, evaluation)| !evaluation.refuted)
                        .collect();
                    if unresolved.is_empty() {
                        let Some(outstanding) = outstanding_group_leaves.get_mut(leaf.group_index)
                        else {
                            return ScalarCoverOutcome::Declined(
                                "internal group index mismatch",
                                stats,
                            );
                        };
                        *outstanding = match outstanding.checked_sub(1) {
                            Some(value) => value,
                            None => {
                                return ScalarCoverOutcome::Declined(
                                    "internal group leaf count underflow",
                                    stats,
                                )
                            }
                        };
                        if *outstanding == 0 {
                            stats.completed_groups = stats.completed_groups.saturating_add(1);
                            let Some(group) = groups.get(leaf.group_index) else {
                                return ScalarCoverOutcome::Declined(
                                    "internal group index mismatch",
                                    stats,
                                );
                            };
                            for &clause_index in &group.clauses {
                                let Some(completed) = completed_clauses.get_mut(clause_index)
                                else {
                                    return ScalarCoverOutcome::Declined(
                                        "internal clause index mismatch",
                                        stats,
                                    );
                                };
                                *completed = true;
                            }
                        }
                        continue;
                    }
                    let Some((lower, upper)) = split_segment(leaf.segment) else {
                        return ScalarCoverOutcome::Declined(
                            "terminal leaf escaped diagnostic scan",
                            stats,
                        );
                    };
                    leaf.obligations = unresolved
                        .into_iter()
                        .map(|(clause_index, _)| clause_index)
                        .collect();
                    let obligations = leaf.obligations.len();
                    let next_frontier_checks = match obligations
                        .checked_mul(2)
                        .and_then(|added| frontier_checks.checked_add(added))
                    {
                        Some(checks) => checks,
                        None => {
                            return ScalarCoverOutcome::Declined(
                                "frontier obligation count overflow",
                                stats,
                            )
                        }
                    };
                    let Some(next_frontier_len) = frontier.len().checked_add(2) else {
                        return ScalarCoverOutcome::Declined("frontier leaf count overflow", stats);
                    };
                    // Check prospective descendants before allocating them.
                    // Every live obligation must be evaluated at least once,
                    // so these are exact lower bounds on unavoidable work.
                    if stats
                        .leaves
                        .checked_add(next_frontier_len)
                        .is_none_or(|minimum| minimum > limits.max_leaves)
                    {
                        return ScalarCoverOutcome::Declined("leaf cap exceeded", stats);
                    }
                    if stats
                        .clause_leaf_checks
                        .checked_add(next_frontier_checks)
                        .is_none_or(|minimum| minimum > limits.max_clause_leaf_checks)
                    {
                        return ScalarCoverOutcome::Declined(
                            "clause-leaf check cap exceeded",
                            stats,
                        );
                    }
                    if next_frontier_len
                        .checked_mul(2)
                        .and_then(|frontier| group_storage.checked_add(frontier))
                        .is_none_or(|stored| stored > limits.max_stored_f64)
                    {
                        return ScalarCoverOutcome::Declined(
                            "f64 boundary-enclosure memory cap exceeded",
                            stats,
                        );
                    }
                    frontier_checks = next_frontier_checks;
                    let Some(outstanding) = outstanding_group_leaves.get_mut(leaf.group_index)
                    else {
                        return ScalarCoverOutcome::Declined(
                            "internal group index mismatch",
                            stats,
                        );
                    };
                    *outstanding = match outstanding.checked_add(1) {
                        Some(value) => value,
                        None => {
                            return ScalarCoverOutcome::Declined(
                                "internal group leaf count overflow",
                                stats,
                            )
                        }
                    };
                    let lower_obligations = leaf.obligations.clone();
                    frontier.push(FrontierLeaf {
                        group_index: leaf.group_index,
                        segment: upper,
                        obligations: leaf.obligations,
                    });
                    frontier.push(FrontierLeaf {
                        group_index: leaf.group_index,
                        segment: lower,
                        obligations: lower_obligations,
                    });
                }
            }
            BatchOutcome::Deadline => return ScalarCoverOutcome::Deadline(stats),
            BatchOutcome::Declined(reason) => return ScalarCoverOutcome::Declined(reason, stats),
        }
        // These are lower bounds on eventual work: every live frontier leaf
        // must be evaluated at least once. Refuse as soon as a cap is
        // inevitably unreachable, before allocating or propagating another
        // batch. The whole partial cover remains non-authoritative.
        if stats
            .leaves
            .checked_add(frontier.len())
            .is_none_or(|minimum| minimum > limits.max_leaves)
        {
            return ScalarCoverOutcome::Declined("leaf cap exceeded", stats);
        }
        if stats
            .clause_leaf_checks
            .checked_add(frontier_checks)
            .is_none_or(|minimum| minimum > limits.max_clause_leaf_checks)
        {
            return ScalarCoverOutcome::Declined("clause-leaf check cap exceeded", stats);
        }
        if frontier
            .len()
            .checked_mul(2)
            .and_then(|frontier| group_storage.checked_add(frontier))
            .is_none_or(|stored| stored > limits.max_stored_f64)
        {
            return ScalarCoverOutcome::Declined(
                "f64 boundary-enclosure memory cap exceeded",
                stats,
            );
        }
    }
    if deadline_expired(deadline) {
        return ScalarCoverOutcome::Deadline(stats);
    }
    debug_assert_eq!(completed_clauses, eligible);
    ScalarCoverOutcome::Verified(stats, eligible)
}

/// Opt-in production seam. By default, only an exact complete cover of the
/// entire admitted 1D subset publishes. A second, explicitly budgeted partial
/// opt-in may publish individual clauses only after their entire group cover
/// retires; every unfinished/authentication-failed clause remains false for the
/// existing screen and downstream verifier.
pub(super) fn try_nn4sys_scalar_complete_cover(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[BTreeMap<usize, (f64, f64)>],
    deadline: Option<Instant>,
) -> Option<Vec<bool>> {
    if !enabled() {
        return None;
    }
    let BetaCrownModel::Graph(graph) = model_net else {
        return None;
    };
    let started = Instant::now();
    let partial = partial_cover_enabled();
    let lane_deadline = if partial {
        partial_cover_deadline(deadline)
    } else {
        deadline
    };
    let mut completed_clauses = vec![false; clauses.len()];
    let outcome = run_complete_scalar_cover_recording(
        graph,
        input,
        clauses,
        per_clause_input_bounds,
        ScalarCoverLimits::from_env(),
        lane_deadline,
        &mut completed_clauses,
    );
    let stats = outcome.stats();
    let result = match outcome {
        ScalarCoverOutcome::Verified(_, proven) => {
            let scope = if stats.eligible_clauses == stats.clauses {
                "whole_property"
            } else {
                "eligible_subset"
            };
            eprintln!(
                "{TELEMETRY_MARKER} outcome=verified scope={scope} groups={} completed_groups={} clauses={} \
                 eligible={} excluded_non_1d={} leaves={} checks={} batches={} depth={} \
                 centered_attempts={} centered_batch_calls={} centered_refuted={} \
                 centered_failures={}",
                stats.groups,
                stats.completed_groups,
                stats.clauses,
                stats.eligible_clauses,
                stats.excluded_non_1d_clauses,
                stats.leaves,
                stats.clause_leaf_checks,
                stats.batches,
                stats.max_depth_reached,
                stats.centered_terminal_attempts,
                stats.centered_terminal_batch_calls,
                stats.centered_terminal_refuted_clauses,
                stats.centered_terminal_failures,
            );
            info!(
                groups = stats.groups,
                completed_groups = stats.completed_groups,
                clauses = stats.clauses,
                eligible_clauses = stats.eligible_clauses,
                excluded_non_1d_clauses = stats.excluded_non_1d_clauses,
                scope,
                leaves = stats.leaves,
                clause_leaf_checks = stats.clause_leaf_checks,
                batches = stats.batches,
                depth = stats.max_depth_reached,
                centered_terminal_attempts = stats.centered_terminal_attempts,
                centered_terminal_batch_calls = stats.centered_terminal_batch_calls,
                centered_terminal_refuted_clauses = stats.centered_terminal_refuted_clauses,
                centered_terminal_failures = stats.centered_terminal_failures,
                elapsed_s = started.elapsed().as_secs_f64(),
                "NN4SYS one-axis f64 complete cover verified its entire eligible scope"
            );
            Some(proven)
        }
        ScalarCoverOutcome::Declined(reason, _) => {
            eprintln!(
                "{TELEMETRY_MARKER} outcome=declined reason={reason:?} groups={} completed_groups={} clauses={} \
                 eligible={} excluded_non_1d={} leaves={} checks={} batches={} depth={} \
                 centered_attempts={} centered_batch_calls={} centered_refuted={} \
                 centered_failures={}",
                stats.groups,
                stats.completed_groups,
                stats.clauses,
                stats.eligible_clauses,
                stats.excluded_non_1d_clauses,
                stats.leaves,
                stats.clause_leaf_checks,
                stats.batches,
                stats.max_depth_reached,
                stats.centered_terminal_attempts,
                stats.centered_terminal_batch_calls,
                stats.centered_terminal_refuted_clauses,
                stats.centered_terminal_failures,
            );
            debug!(
                reason,
                leaves = stats.leaves,
                elapsed_s = started.elapsed().as_secs_f64(),
                "NN4SYS one-axis f64 complete cover declined; preserving existing path"
            );
            None
        }
        ScalarCoverOutcome::Incomplete(_, diagnostics) => {
            let elapsed_s = started.elapsed().as_secs_f64();
            eprintln!(
                "{TELEMETRY_MARKER} outcome=incomplete groups={} completed_groups={} clauses={} eligible={} \
                 excluded_non_1d={} leaves={} checks={} batches={} depth={} unresolved={} \
                 centered_attempts={} centered_batch_calls={} centered_refuted={} \
                 centered_failures={} \
                 elapsed_s={elapsed_s:.6}",
                stats.groups,
                stats.completed_groups,
                stats.clauses,
                stats.eligible_clauses,
                stats.excluded_non_1d_clauses,
                stats.leaves,
                stats.clause_leaf_checks,
                stats.batches,
                stats.max_depth_reached,
                diagnostics.len(),
                stats.centered_terminal_attempts,
                stats.centered_terminal_batch_calls,
                stats.centered_terminal_refuted_clauses,
                stats.centered_terminal_failures,
            );
            for diagnostic in &diagnostics {
                let segment_lower = f64::from_bits(diagnostic.segment_lower_bits);
                let segment_upper = f64::from_bits(diagnostic.segment_upper_bits);
                let margin = f64::from_bits(diagnostic.refutation_margin_bits);
                eprintln!(
                    "{TELEMETRY_MARKER} outcome=incomplete_detail group={} clause={} \
                     free_axis={} segment_lower={segment_lower:.17e} \
                     segment_upper={segment_upper:.17e} \
                     segment_lower_bits=0x{:016x} segment_upper_bits=0x{:016x} depth={} \
                     best_constraint={} refutation_margin={margin:.17e} \
                     refutation_margin_bits=0x{:016x} equality_refutes={}",
                    diagnostic.group_index,
                    diagnostic.clause_index,
                    diagnostic.free_axis,
                    diagnostic.segment_lower_bits,
                    diagnostic.segment_upper_bits,
                    diagnostic.depth,
                    diagnostic.best_constraint_index,
                    diagnostic.refutation_margin_bits,
                    diagnostic.equality_refutes,
                );
            }
            debug!(
                leaves = stats.leaves,
                unresolved = diagnostics.len(),
                elapsed_s,
                "NN4SYS one-axis f64 complete cover found an undecided leaf"
            );
            None
        }
        ScalarCoverOutcome::Deadline(_) => {
            eprintln!(
                "{TELEMETRY_MARKER} outcome=deadline groups={} completed_groups={} clauses={} eligible={} \
                 excluded_non_1d={} leaves={} checks={} batches={} depth={} \
                 centered_attempts={} centered_batch_calls={} centered_refuted={} \
                 centered_failures={}",
                stats.groups,
                stats.completed_groups,
                stats.clauses,
                stats.eligible_clauses,
                stats.excluded_non_1d_clauses,
                stats.leaves,
                stats.clause_leaf_checks,
                stats.batches,
                stats.max_depth_reached,
                stats.centered_terminal_attempts,
                stats.centered_terminal_batch_calls,
                stats.centered_terminal_refuted_clauses,
                stats.centered_terminal_failures,
            );
            debug!(
                leaves = stats.leaves,
                elapsed_s = started.elapsed().as_secs_f64(),
                "NN4SYS one-axis f64 complete cover stopped at deadline"
            );
            None
        }
    };
    if result.is_none() && partial {
        let completed = completed_clauses.iter().filter(|&&value| value).count();
        if completed > 0 {
            eprintln!(
                "{TELEMETRY_MARKER} outcome=partial completed_clauses={completed} \
                 total_clauses={} completed_groups={} elapsed_s={:.6}",
                clauses.len(),
                stats.completed_groups,
                started.elapsed().as_secs_f64(),
            );
            return Some(completed_clauses);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_propagate::{GraphNode, Layer};

    fn boxed(lower: f32, upper: f32) -> BoundedTensor {
        BoundedTensor::new(
            ndarray::arr1(&[lower]).into_dyn(),
            ndarray::arr1(&[upper]).into_dyn(),
        )
        .unwrap()
    }

    fn boxed_shape(shape: &[usize], lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(shape), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(ndarray::IxDyn(shape), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    /// y = relu(x) - relu(x).  Interval dependency leaves `[-w, w]` on a
    /// positive leaf of width `w`, while the concrete function is zero.
    fn cancellation_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "relu",
            Layer::ReLU(ny_propagate::layers::ReLULayer),
        ));
        graph.add_node(GraphNode::binary(
            "out",
            Layer::Sub(ny_propagate::layers::SubLayer),
            "relu",
            "relu",
        ));
        graph.set_output("out");
        graph
    }

    fn one_box(lower: f64, upper: f64) -> BTreeMap<usize, (f64, f64)> {
        BTreeMap::from([(0, (lower, upper))])
    }

    fn full_box(bounds: &[(f64, f64)]) -> BTreeMap<usize, (f64, f64)> {
        bounds.iter().copied().enumerate().collect()
    }

    fn limits(depth: u8) -> ScalarCoverLimits {
        ScalarCoverLimits {
            depth,
            max_leaves: 1024,
            max_clause_leaf_checks: 2048,
            max_batch_leaves: 16,
            max_stored_f64: 4096,
            centered_terminal: false,
            centered_start_depth: depth,
            centered_batch_leaves: DEFAULT_CENTERED_TERMINAL_BATCH_LEAVES,
        }
    }

    #[test]
    fn dyadic_cover_is_exact_deterministic_and_lower_first() {
        let leaves: Vec<_> = DyadicLeaves::new(-1.0, 1.0, 3).collect();
        assert_eq!(leaves.len(), 8);
        assert_eq!(leaves.first().unwrap().lower, -1.0);
        assert_eq!(leaves.last().unwrap().upper, 1.0);
        assert!(leaves
            .windows(2)
            .all(|pair| pair[0].upper.to_bits() == pair[1].lower.to_bits()));
        assert!(leaves.iter().all(|leaf| leaf.depth == 3));

        let adjacent_upper = 1.0f64.next_up();
        let adjacent: Vec<_> = DyadicLeaves::new(1.0, adjacent_upper, 24).collect();
        assert_eq!(
            adjacent,
            vec![Segment {
                lower: 1.0,
                upper: adjacent_upper,
                depth: 0,
            }],
            "an unsplittable parent must remain one exact leaf"
        );
    }

    #[test]
    fn complete_cover_proves_only_after_every_group_leaf_is_safe() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        // At depth 0, IBP gives upper ~= 1 and cannot refute Y >= 0.26.
        // At depth 2, every leaf has width 0.25, so all four refute it.
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.26)]];
        let boxes = vec![one_box(0.0, 1.0)];
        let ScalarCoverOutcome::Incomplete(_, diagnostics) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, limits(0), None)
        else {
            panic!("the root enclosure should remain open");
        };
        assert_eq!(diagnostics.len(), 1);
        let diagnostic = &diagnostics[0];
        assert_eq!(diagnostic.group_index, 0);
        assert_eq!(diagnostic.clause_index, 0);
        assert_eq!(diagnostic.free_axis, 0);
        assert_eq!(diagnostic.segment_lower_bits, 0.0f64.to_bits());
        assert_eq!(diagnostic.segment_upper_bits, 1.0f64.to_bits());
        assert_eq!(diagnostic.depth, 0);
        assert_eq!(diagnostic.best_constraint_index, 0);
        assert!(f64::from_bits(diagnostic.refutation_margin_bits) < 0.0);
        assert!(!diagnostic.equality_refutes);
        let ScalarCoverOutcome::Verified(stats, proven) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, limits(2), None)
        else {
            panic!("the four-leaf complete cover should verify");
        };
        assert_eq!(stats.groups, 1);
        assert_eq!(stats.completed_groups, 1);
        assert_eq!(stats.clauses, 1);
        assert_eq!(proven, vec![true]);
        assert_eq!(
            stats.leaves, 7,
            "adaptive cover evaluates root, two children, then four leaves"
        );
        assert_eq!(stats.clause_leaf_checks, 7);
        assert_eq!(stats.max_depth_reached, 2);
    }

    #[test]
    fn partial_recording_marks_only_a_fully_retired_group() {
        let graph = cancellation_graph();
        let input = boxed_shape(&[2], &[0.0, 0.0], &[1.0, 1.0]);
        let clauses = vec![
            vec![OutputConstraint::GreaterEqConst(0, 1.1)],
            vec![OutputConstraint::GreaterEqConst(0, 0.0)],
        ];
        let boxes = vec![
            full_box(&[(0.0, 1.0), (0.0, 0.0)]),
            full_box(&[(0.0, 1.0), (0.5, 0.5)]),
        ];
        let mut completed = vec![false; clauses.len()];
        let ScalarCoverOutcome::Incomplete(stats, _) = run_complete_scalar_cover_recording(
            &graph,
            &input,
            &clauses,
            &boxes,
            limits(0),
            None,
            &mut completed,
        ) else {
            panic!("the second group's satisfiable atom must keep the run incomplete");
        };
        assert_eq!(stats.groups, 2);
        assert_eq!(stats.completed_groups, 1);
        assert_eq!(completed, vec![true, false]);

        let model = BetaCrownModel::Graph(Box::new(graph));
        let published = ny_test_utils::env::with_serialized_env_vars(
            &[
                ("NY_NN4SYS_1D_COMPLETE_COVER", "1"),
                ("NY_NN4SYS_1D_PARTIAL_COVER", "1"),
                ("NY_NN4SYS_1D_COVER_DEPTH", "0"),
                ("NY_NN4SYS_1D_PARTIAL_BUDGET_MS", "1000"),
            ],
            || try_nn4sys_scalar_complete_cover(&model, &input, &clauses, &boxes, None),
        );
        assert_eq!(published, Some(vec![true, false]));
    }

    #[test]
    fn exact_opt_in_centered_terminal_can_close_a_dependency_leaf() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.26)]];
        let boxes = vec![one_box(0.0, 1.0)];
        let mut centered = limits(0);
        centered.centered_terminal = true;
        let ScalarCoverOutcome::Verified(stats, proven) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, centered, None)
        else {
            panic!("the rigorous centered terminal enclosure should cancel relu(x)-relu(x)");
        };
        assert_eq!(proven, vec![true]);
        assert_eq!(stats.leaves, 1);
        assert_eq!(stats.centered_terminal_attempts, 1);
        assert_eq!(stats.centered_terminal_refuted_clauses, 1);
        assert_eq!(stats.centered_terminal_failures, 0);
    }

    #[test]
    fn centered_start_depth_prunes_before_the_hard_cover_depth() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.26)]];
        let boxes = vec![one_box(0.0, 1.0)];
        let mut adaptive = limits(2);
        adaptive.centered_terminal = true;
        adaptive.centered_start_depth = 0;
        let ScalarCoverOutcome::Verified(stats, proven) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, adaptive, None)
        else {
            panic!("a centered sibling batch should prune before the hard cover depth");
        };
        assert_eq!(proven, vec![true]);
        assert_eq!(stats.leaves, 3, "root plus two depth-one siblings");
        assert_eq!(stats.max_depth_reached, 1);
        assert_eq!(stats.centered_terminal_attempts, 2);
        assert_eq!(stats.centered_terminal_batch_calls, 1);
    }

    #[test]
    fn centered_start_depth_keeps_splitting_an_open_leaf() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.0)]];
        let boxes = vec![one_box(0.0, 1.0)];
        let mut adaptive = limits(2);
        adaptive.centered_terminal = true;
        adaptive.centered_start_depth = 0;
        let ScalarCoverOutcome::Incomplete(stats, diagnostics) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, adaptive, None)
        else {
            panic!("a satisfiable atom must stay open through the hard cover depth");
        };
        assert_eq!(stats.max_depth_reached, 2);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].depth, 2);
        assert!(stats.centered_terminal_attempts >= 4);
        assert_eq!(stats.centered_terminal_refuted_clauses, 0);
    }

    #[test]
    fn centered_terminal_batches_reachable_sibling_cells_once() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.26)]];
        let boxes = vec![one_box(0.0, 1.0)];
        let mut centered = limits(1);
        centered.centered_terminal = true;
        let ScalarCoverOutcome::Verified(stats, proven) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, centered, None)
        else {
            panic!("one centered batch should close both terminal dyadic siblings");
        };
        assert_eq!(proven, vec![true]);
        assert_eq!(stats.leaves, 3, "root plus both depth-one children");
        assert_eq!(stats.centered_terminal_attempts, 2);
        assert_eq!(stats.centered_terminal_batch_calls, 1);
        assert_eq!(stats.centered_terminal_refuted_clauses, 2);
        assert_eq!(stats.centered_terminal_failures, 0);
    }

    #[test]
    fn centered_terminal_batch_accounts_for_selected_input_clones() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.26)]];
        let boxes = vec![one_box(0.0, 1.0)];
        let mut centered = limits(1);
        centered.centered_terminal = true;

        // At the two-child wave: six retained group/frontier words, four
        // original batch-input words, four selected-input clone words, and
        // twelve value/centered/mono output words total exactly 26.
        centered.max_stored_f64 = 25;
        let ScalarCoverOutcome::Verified(stats, _) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, centered, None)
        else {
            panic!("the memory-gated per-cell fallback should remain complete");
        };
        assert_eq!(stats.centered_terminal_batch_calls, 0);
        assert_eq!(stats.centered_terminal_attempts, 2);

        centered.max_stored_f64 = 26;
        let ScalarCoverOutcome::Verified(stats, _) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, centered, None)
        else {
            panic!("the exactly-accounted two-cell centered batch should fit");
        };
        assert_eq!(stats.centered_terminal_batch_calls, 1);
        assert_eq!(stats.centered_terminal_attempts, 2);
    }

    #[test]
    fn centered_terminal_keeps_a_satisfiable_equality_open() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.0)]];
        let boxes = vec![one_box(0.0, 1.0)];
        let mut centered = limits(0);
        centered.centered_terminal = true;
        let ScalarCoverOutcome::Incomplete(stats, diagnostics) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, centered, None)
        else {
            panic!("Y=0 satisfies the non-strict unsafe atom and must remain open");
        };
        assert_eq!(stats.centered_terminal_attempts, 1);
        assert_eq!(stats.centered_terminal_refuted_clauses, 0);
        assert_eq!(diagnostics.len(), 1);
        assert!(!diagnostics[0].equality_refutes);
        assert!(f64::from_bits(diagnostics[0].refutation_margin_bits) <= 0.0);
    }

    #[test]
    fn centered_terminal_accounts_for_all_returned_output_surfaces() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.26)]];
        let boxes = vec![one_box(0.0, 1.0)];
        let mut centered = limits(0);
        centered.centered_terminal = true;
        // Four retained group/frontier words + two input words + six
        // value/centered/mono output words require twelve f64-sized slots.
        centered.max_stored_f64 = 11;
        let ScalarCoverOutcome::Incomplete(stats, _) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, centered, None)
        else {
            panic!("the centered fallback must stay dark when its surfaces do not fit");
        };
        assert_eq!(stats.centered_terminal_attempts, 0);

        centered.max_stored_f64 = 12;
        let ScalarCoverOutcome::Verified(stats, proven) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, centered, None)
        else {
            panic!("the exactly-accounted centered fallback should fit");
        };
        assert_eq!(proven, vec![true]);
        assert_eq!(stats.centered_terminal_attempts, 1);
    }

    #[test]
    fn centered_terminal_absorbs_an_outward_authored_point_without_seeding_it() {
        let graph = cancellation_graph();
        let input = boxed_shape(&[2], &[-1.0, -1.0], &[1.0, 1.0]);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.26)]];
        let boxes = vec![full_box(&[(0.0, 1.0), (0.1, 0.1)])];
        let mut centered = limits(0);
        centered.centered_terminal = true;
        let ScalarCoverOutcome::Verified(stats, proven) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, centered, None)
        else {
            panic!("the full outward point enclosure belongs in the sound center-box channel");
        };
        assert_eq!(proven, vec![true]);
        assert_eq!(stats.centered_terminal_attempts, 1);
        assert_eq!(stats.centered_terminal_refuted_clauses, 1);

        let two_seeded_axes = Interval64 {
            lower: ndarray::arr1(&[0.0, 0.0]).into_dyn(),
            upper: ndarray::arr1(&[1.0, 1.0]).into_dyn(),
        };
        assert!(
            !graph.ibp_f64_centered_only_seeds_axis(&two_seeded_axes, 0),
            "a genuinely second derivative axis must fail the one-axis bridge"
        );
    }

    #[test]
    fn identical_clause_domains_share_walks_but_keep_clause_obligations() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        let clauses = vec![
            vec![OutputConstraint::GreaterEqConst(0, 0.26)],
            vec![OutputConstraint::LessThanConst(0, -0.26)],
        ];
        let boxes = vec![one_box(0.0, 1.0), one_box(0.0, 1.0)];
        let ScalarCoverOutcome::Verified(stats, proven) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, limits(2), None)
        else {
            panic!("both band sides should be refuted on every shared leaf");
        };
        assert_eq!(stats.groups, 1);
        assert_eq!(proven, vec![true, true]);
        assert_eq!(stats.leaves, 7, "the shared input cover is evaluated once");
        assert_eq!(stats.clause_leaf_checks, 14);
    }

    #[test]
    fn parent_refuted_obligations_do_not_descend_with_harder_siblings() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        let clauses = vec![
            // The root output [-1, 1] already refutes Y >= 1.1.
            vec![OutputConstraint::GreaterEqConst(0, 1.1)],
            // This sibling needs all four depth-two leaves.
            vec![OutputConstraint::GreaterEqConst(0, 0.26)],
        ];
        let boxes = vec![one_box(0.0, 1.0), one_box(0.0, 1.0)];
        let ScalarCoverOutcome::Verified(stats, proven) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, limits(2), None)
        else {
            panic!(
                "ancestor certificates plus descendant leaves should exactly cover both clauses"
            );
        };
        assert_eq!(stats.groups, 1);
        assert_eq!(proven, vec![true, true]);
        assert_eq!(stats.leaves, 7);
        assert_eq!(
            stats.clause_leaf_checks, 8,
            "two root checks plus one unresolved check on each of six descendants"
        );
    }

    #[test]
    fn strict_constraint_falsity_is_correct_at_equal_boundaries() {
        let lower = [1.0, 1.0];
        let upper = [1.0, 1.0];

        assert!(!constraint_provably_false(
            &lower,
            &upper,
            &OutputConstraint::LessEqConst(0, 1.0)
        ));
        assert!(constraint_provably_false(
            &lower,
            &upper,
            &OutputConstraint::LessThanConst(0, 1.0)
        ));
        assert!(!constraint_provably_false(
            &lower,
            &upper,
            &OutputConstraint::GreaterEqConst(0, 1.0)
        ));
        assert!(constraint_provably_false(
            &lower,
            &upper,
            &OutputConstraint::GreaterThanConst(0, 1.0)
        ));

        assert!(!constraint_provably_false(
            &lower,
            &upper,
            &OutputConstraint::LessEq(0, 1)
        ));
        assert!(constraint_provably_false(
            &lower,
            &upper,
            &OutputConstraint::LessThan(0, 1)
        ));
        assert!(!constraint_provably_false(
            &lower,
            &upper,
            &OutputConstraint::GreaterEq(0, 1)
        ));
        assert!(constraint_provably_false(
            &lower,
            &upper,
            &OutputConstraint::GreaterThan(0, 1)
        ));
    }

    #[test]
    fn nonstandard_output_layout_fails_open_instead_of_reindexing() {
        let lower = ndarray::arr2(&[[0.0, 1.0], [2.0, 3.0]])
            .reversed_axes()
            .into_dyn();
        let upper = lower.clone();
        assert!(standard_output_slices(&Interval64 { lower, upper }).is_none());
    }

    #[test]
    fn one_unproved_clause_discards_the_whole_cover() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        let clauses = vec![
            vec![OutputConstraint::GreaterEqConst(0, 0.26)],
            // Y >= 0 is actually satisfiable, so no leaf may refute it.
            vec![OutputConstraint::GreaterEqConst(0, 0.0)],
        ];
        let boxes = vec![one_box(0.0, 1.0), one_box(0.0, 1.0)];
        let ScalarCoverOutcome::Incomplete(_, diagnostics) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, limits(2), None)
        else {
            panic!("the satisfiable sibling must keep the atomic cover open");
        };
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].group_index, 0);
        assert_eq!(diagnostics[0].clause_index, 1);
        assert_eq!(diagnostics[0].depth, 2);
    }

    #[test]
    fn caps_and_deadline_never_publish_partial_coverage() {
        let graph = cancellation_graph();
        let input = boxed(0.0, 1.0);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.26)]];
        let boxes = vec![one_box(0.0, 1.0)];

        let mut capped = limits(4);
        capped.max_leaves = 3;
        assert!(matches!(
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, capped, None),
            ScalarCoverOutcome::Declined("leaf cap exceeded", _)
        ));

        let mut depth_capped = limits(2);
        depth_capped.depth = HARD_MAX_DEPTH + 1;
        assert!(matches!(
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, depth_capped, None),
            ScalarCoverOutcome::Declined("invalid cover limits", _)
        ));

        let mut memory_capped = limits(2);
        // The retained group root plus frontier root fit in four f64s, but
        // the duplicated Interval64 input and scalar output need four more.
        memory_capped.max_stored_f64 = 4;
        assert!(matches!(
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, memory_capped, None),
            ScalarCoverOutcome::Declined("f64 boundary-enclosure memory cap admits no leaf", _)
        ));

        let mut check_capped = limits(2);
        check_capped.max_clause_leaf_checks = 1;
        assert!(matches!(
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, check_capped, None),
            ScalarCoverOutcome::Declined("clause-leaf check cap exceeded", _)
        ));

        assert!(matches!(
            run_complete_scalar_cover(
                &graph,
                &input,
                &clauses,
                &boxes,
                limits(2),
                Some(Instant::now()),
            ),
            ScalarCoverOutcome::Deadline(_)
        ));
    }

    #[test]
    fn authored_endpoints_are_outward_rounded_before_f64_cover() {
        let input = boxed(-1.0, 1.0);
        let lower = 0.1f64;
        let upper = 0.2f64;
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 2.0)]];
        let boxes = vec![one_box(lower, upper)];
        let admission =
            build_clause_groups(&input, &clauses, &boxes, limits(0).max_stored_f64, None).unwrap();
        assert_eq!(admission.groups.len(), 1);
        let group = &admission.groups[0];
        let group_lower = group.complete_box.lower(0).unwrap();
        let group_upper = group.complete_box.upper(0).unwrap();
        assert_eq!(group_lower, f64::from(f64_to_f32_down(lower)));
        assert_eq!(group_upper, f64::from(f64_to_f32_up(upper)));
        assert!(group_lower <= lower);
        assert!(group_upper >= upper);
    }

    #[test]
    fn complete_box_admission_rejects_omitted_or_unordered_coordinates() {
        let input = boxed_shape(&[1, 3], &[0.0, -1.0, -2.0], &[1.0, 1.0, 2.0]);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 2.0)]];

        // X_2 remains globally unfixed. Omitting it must not let admission
        // silently manufacture a point value.
        let omitted = vec![full_box(&[(0.0, 1.0), (0.0, 0.0)])];
        assert!(matches!(
            build_clause_groups(&input, &clauses, &omitted, limits(0).max_stored_f64, None,),
            Err(AdmissionError::Declined(
                "clause box does not completely author every input coordinate"
            ))
        ));

        let mut shifted = BTreeMap::new();
        shifted.insert(0, (0.0, 1.0));
        shifted.insert(1, (0.0, 0.0));
        shifted.insert(3, (0.0, 0.0));
        assert!(matches!(
            build_clause_groups(&input, &clauses, &[shifted], limits(0).max_stored_f64, None,),
            Err(AdmissionError::Declined(
                "clause box does not completely author every input coordinate"
            ))
        ));
    }

    #[test]
    fn nonrepresentable_authored_points_remain_outward_enclosed() {
        let input = boxed_shape(&[1, 2], &[-1.0, -1.0], &[1.0, 1.0]);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 2.0)]];
        let fixed = 0.1f64;
        let boxes = vec![full_box(&[(0.0, 1.0), (fixed, fixed)])];
        let admission =
            build_clause_groups(&input, &clauses, &boxes, limits(0).max_stored_f64, None).unwrap();
        let complete_box = &admission.groups[0].complete_box;
        let fixed_lower = complete_box.lower(1).unwrap();
        let fixed_upper = complete_box.upper(1).unwrap();
        assert_eq!(fixed_lower, f64::from(f64_to_f32_down(fixed)));
        assert_eq!(fixed_upper, f64::from(f64_to_f32_up(fixed)));
        assert!(fixed_lower <= fixed && fixed <= fixed_upper);
        assert!(
            fixed_lower < fixed_upper,
            "the authored point is not f32-exact"
        );
    }

    #[test]
    fn full_shape_leaf_changes_only_its_authenticated_free_axis() {
        let shape = [1, 2, 2];
        let input = boxed_shape(&shape, &[-1.0; 4], &[1.0; 4]);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(2, 2.0)]];
        let boxes = vec![full_box(&[
            (-0.5, -0.5),
            (0.1, 0.1),
            (0.0, 1.0),
            (0.25, 0.25),
        ])];
        let admission =
            build_clause_groups(&input, &clauses, &boxes, limits(0).max_stored_f64, None).unwrap();
        let group = &admission.groups[0];
        assert_eq!(group.complete_box.free_axis, 2);
        let segment = Segment {
            lower: 0.25,
            upper: 0.5,
            depth: 2,
        };
        let leaf = one_axis_input_interval(&input, group, segment).unwrap();
        assert_eq!(leaf.lower.shape(), shape);
        assert_eq!(leaf.upper.shape(), shape);
        let lower = leaf.lower.as_slice().unwrap();
        let upper = leaf.upper.as_slice().unwrap();
        assert_eq!((lower[2], upper[2]), (0.25, 0.5));
        for axis in [0usize, 1, 3] {
            assert_eq!(
                lower[axis].to_bits(),
                group.complete_box.lower(axis).unwrap().to_bits()
            );
            assert_eq!(
                upper[axis].to_bits(),
                group.complete_box.upper(axis).unwrap().to_bits()
            );
        }
    }

    #[test]
    fn groups_may_vary_different_axes_and_never_merge_distinct_fixed_templates() {
        let graph = cancellation_graph();
        let input = boxed_shape(&[2], &[0.0, 0.0], &[1.0, 1.0]);
        let clauses = vec![
            vec![OutputConstraint::GreaterEqConst(0, 0.26)],
            vec![OutputConstraint::GreaterEqConst(1, 0.26)],
            vec![OutputConstraint::LessThanConst(0, -0.26)],
        ];
        let boxes = vec![
            full_box(&[(0.0, 1.0), (0.0, 0.0)]),
            full_box(&[(0.0, 0.0), (0.0, 1.0)]),
            // Same free axis/range as clause 0, but a distinct fixed X_1.
            full_box(&[(0.0, 1.0), (0.5, 0.5)]),
        ];
        let ScalarCoverOutcome::Verified(stats, proven) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, limits(2), None)
        else {
            panic!("all three exact one-axis boxes should verify");
        };
        assert_eq!(
            stats.groups, 3,
            "full fixed-coordinate templates must key groups"
        );
        assert_eq!(stats.completed_groups, 3);
        assert_eq!(stats.eligible_clauses, 3);
        assert_eq!(proven, vec![true, true, true]);
    }

    #[test]
    fn authenticated_two_axis_clause_is_never_claimed() {
        let graph = cancellation_graph();
        let input = boxed_shape(&[2], &[0.0, 0.0], &[1.0, 1.0]);
        let clauses = vec![
            vec![OutputConstraint::GreaterEqConst(0, 0.26)],
            vec![OutputConstraint::GreaterEqConst(0, 100.0)],
        ];
        let boxes = vec![
            full_box(&[(0.0, 1.0), (0.0, 0.0)]),
            full_box(&[(0.0, 1.0), (0.0, 1.0)]),
        ];
        let ScalarCoverOutcome::Verified(stats, proven) =
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, limits(2), None)
        else {
            panic!("the entire eligible 1D subset should verify");
        };
        assert_eq!(stats.eligible_clauses, 1);
        assert_eq!(stats.excluded_non_1d_clauses, 1);
        assert_eq!(proven, vec![true, false]);

        let model = BetaCrownModel::Graph(Box::new(graph));
        let published = ny_test_utils::env::with_serialized_env_vars(
            &[
                ("NY_NN4SYS_1D_COMPLETE_COVER", "1"),
                ("NY_NN4SYS_1D_COVER_DEPTH", "2"),
            ],
            || try_nn4sys_scalar_complete_cover(&model, &input, &clauses, &boxes, None),
        );
        assert_eq!(
            published,
            Some(vec![true, false]),
            "the production seam may publish the atomically complete 1D subset, never the 2D clause"
        );
    }

    #[test]
    fn full_shape_memory_cap_counts_every_coordinate_boundary() {
        let graph = cancellation_graph();
        let input = boxed_shape(&[3], &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 2.0)]];
        let boxes = vec![full_box(&[(0.0, 1.0), (0.0, 0.0), (0.0, 0.0)])];
        let mut capped = limits(0);
        // Six retained full-box words + two frontier endpoints fit. A leaf
        // needs six input boundary values plus at least two output values.
        capped.max_stored_f64 = 15;
        assert!(matches!(
            run_complete_scalar_cover(&graph, &input, &clauses, &boxes, capped, None),
            ScalarCoverOutcome::Declined("f64 boundary-enclosure memory cap admits no leaf", _)
        ));
    }

    #[test]
    fn opt_in_seam_publishes_only_the_whole_property_verdict() {
        let model = BetaCrownModel::Graph(Box::new(cancellation_graph()));
        let input = boxed(0.0, 1.0);
        let clauses = vec![
            vec![OutputConstraint::GreaterEqConst(0, 0.26)],
            vec![OutputConstraint::LessThanConst(0, -0.26)],
        ];
        let boxes = vec![one_box(0.0, 1.0), one_box(0.0, 1.0)];
        let result = ny_test_utils::env::with_serialized_env_vars(
            &[
                ("NY_NN4SYS_1D_COMPLETE_COVER", "1"),
                ("NY_NN4SYS_1D_COVER_DEPTH", "2"),
            ],
            || try_nn4sys_scalar_complete_cover(&model, &input, &clauses, &boxes, None),
        );
        assert_eq!(result, Some(vec![true, true]));

        let incomplete = vec![
            vec![OutputConstraint::GreaterEqConst(0, 0.26)],
            vec![OutputConstraint::GreaterEqConst(0, 0.0)],
        ];
        let result = ny_test_utils::env::with_serialized_env_vars(
            &[
                ("NY_NN4SYS_1D_COMPLETE_COVER", "1"),
                ("NY_NN4SYS_1D_COVER_DEPTH", "2"),
            ],
            || try_nn4sys_scalar_complete_cover(&model, &input, &incomplete, &boxes, None),
        );
        assert_eq!(
            result, None,
            "one open clause must suppress all publication"
        );
    }

    #[test]
    fn production_seam_is_exactly_default_off() {
        let model = BetaCrownModel::Graph(Box::new(cancellation_graph()));
        let input = boxed(0.0, 1.0);
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 1.1)]];
        let boxes = vec![one_box(0.0, 1.0)];
        let result = ny_test_utils::env::with_serialized_env_vars(
            &[("NY_NN4SYS_1D_COMPLETE_COVER", "0")],
            || try_nn4sys_scalar_complete_cover(&model, &input, &clauses, &boxes, None),
        );
        assert_eq!(result, None);
    }

    #[test]
    fn centered_terminal_is_a_second_exact_opt_in() {
        let disabled = ny_test_utils::env::with_serialized_env_vars(
            &[("NY_NN4SYS_1D_CENTERED_TERMINAL", "0")],
            || ScalarCoverLimits::from_env().centered_terminal,
        );
        assert!(!disabled);
        let enabled = ny_test_utils::env::with_serialized_env_vars(
            &[("NY_NN4SYS_1D_CENTERED_TERMINAL", "1")],
            || ScalarCoverLimits::from_env().centered_terminal,
        );
        assert!(enabled);
        let near_miss = ny_test_utils::env::with_serialized_env_vars(
            &[("NY_NN4SYS_1D_CENTERED_TERMINAL", "true")],
            || ScalarCoverLimits::from_env().centered_terminal,
        );
        assert!(!near_miss);

        let adaptive_start = ny_test_utils::env::with_serialized_env_vars(
            &[
                ("NY_NN4SYS_1D_COVER_DEPTH", "16"),
                ("NY_NN4SYS_1D_CENTERED_START_DEPTH", "12"),
            ],
            ScalarCoverLimits::from_env,
        );
        assert_eq!(adaptive_start.depth, 16);
        assert_eq!(adaptive_start.centered_start_depth, 12);
        assert!(adaptive_start.valid());

        let centered_batch = ny_test_utils::env::with_serialized_env_vars(
            &[("NY_NN4SYS_1D_CENTERED_BATCH", "32")],
            ScalarCoverLimits::from_env,
        );
        assert_eq!(centered_batch.centered_batch_leaves, 32);

        let inverted = ny_test_utils::env::with_serialized_env_vars(
            &[
                ("NY_NN4SYS_1D_COVER_DEPTH", "12"),
                ("NY_NN4SYS_1D_CENTERED_START_DEPTH", "16"),
            ],
            ScalarCoverLimits::from_env,
        );
        assert!(!inverted.valid());
    }

    #[test]
    #[cfg(feature = "external-vnncomp")]
    fn authentic_mscn_2048_dual_property_admission_receipt() {
        let property = std::env::var("NY_NN4SYS_1D_PROBE_VNNLIB")
            .expect("set NY_NN4SYS_1D_PROBE_VNNLIB to the authentic property");
        let spec = ny_onnx::vnnlib::load_vnnlib(property).unwrap();
        assert_eq!(spec.num_inputs, 308);
        assert_eq!(spec.output_constraint_clauses.len(), 240);
        let (lower, upper) = spec.split_input_bounds_f32();
        // The authentic ONNX input is [1, 22, 14]; GraphNetwork removes its
        // leading batch dimension before verification.
        let input = boxed_shape(&[22, 14], &lower, &upper);
        let admission = build_clause_groups(
            &input,
            &spec.output_constraint_clauses,
            &spec.per_clause_input_bounds,
            HARD_MAX_STORED_F64,
            None,
        )
        .unwrap();
        let eligible = admission
            .eligible
            .iter()
            .filter(|&&eligible| eligible)
            .count();
        let mut free_axes: Vec<_> = admission
            .groups
            .iter()
            .map(|group| group.complete_box.free_axis)
            .collect();
        free_axes.sort_unstable();
        free_axes.dedup();
        eprintln!(
            "{TELEMETRY_MARKER} authentic_admission=true clauses={} eligible={} \
             excluded_non_1d={} groups={} input_elements={} free_axes={free_axes:?}",
            spec.output_constraint_clauses.len(),
            eligible,
            spec.output_constraint_clauses.len() - eligible,
            admission.groups.len(),
            input.lower().len(),
        );
        assert_eq!(eligible, 182);
        assert_eq!(spec.output_constraint_clauses.len() - eligible, 58);
        assert_eq!(free_axes, vec![205, 207, 219, 221, 233, 235]);
    }
}
